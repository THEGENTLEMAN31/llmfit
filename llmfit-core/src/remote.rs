//! Remote GGUF header inspection over HTTP range requests.
//!
//! HuggingFace resolve URLs (`https://huggingface.co/{repo}/resolve/main/{file}`)
//! redirect to a CDN that honors `Range` headers. [`RangeReader`] exploits
//! this to stream only the bytes a GGUF header actually needs: sequential
//! chunks are fetched on demand, and skipped payloads (tokenizer arrays) are
//! jumped over without ever hitting the wire. The transfer budget is capped
//! so a pathological file can never pull model weights.

use std::io::{self, Read, Seek, SeekFrom};
use std::time::Duration;

use crate::gguf::{GgufHeader, GgufModelSummary};

/// Bytes fetched per range request. The window doubles after every fetch
/// (up to [`MAX_CHUNK_BYTES`]) so even multi-MiB tokenizer metadata needs
/// only a couple of round-trips (measured: ~0.9 s fixed cost per request
/// against the HF CDN, ~6 MB/s sustained).
pub const INITIAL_CHUNK_BYTES: u64 = 4 * 1024 * 1024;
pub const MAX_CHUNK_BYTES: u64 = 16 * 1024 * 1024;
/// Hard ceiling on total bytes transferred while parsing one header.
///
/// This guards against ever pulling model weights, not against metadata
/// itself: string-array metadata (tokenizer `tokens`/`merges`) interleaves
/// length prefixes through megabytes of payload and cannot be range-skipped,
/// so real headers legitimately reach several MiB (measured 7.3 MiB for
/// Llama-3.2-1B). Fixed-width arrays and tensor bodies ARE skipped via
/// seeks and stay off the wire.
pub const DEFAULT_TRANSFER_CAP_BYTES: u64 = 32 * 1024 * 1024;

const REQUEST_TIMEOUT_SECS: u64 = 30;
const MAX_REDIRECTS: usize = 5;

/// A lazy `Read + Seek` view over a remote file served with byte-range
/// support. Seeks never fetch; the next read after a jump outside the
/// buffered window issues exactly one range request for the chunk containing
/// the target offset.
pub struct RangeReader {
    url: String,
    /// CDN URL captured after the first redirect chain, reused verbatim.
    resolved_url: Option<String>,
    next_chunk: u64,
    transfer_cap: u64,
    buf: Vec<u8>,
    /// Absolute offset of `buf[0]`; `None` when the buffer is empty/invalid.
    buf_start: Option<u64>,
    pos: u64,
    transferred: u64,
    total_size: Option<u64>,
    last_fatal: Option<String>,
}

impl RangeReader {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            resolved_url: None,
            next_chunk: INITIAL_CHUNK_BYTES,
            transfer_cap: DEFAULT_TRANSFER_CAP_BYTES,
            buf: Vec::new(),
            buf_start: None,
            pos: 0,
            transferred: 0,
            total_size: None,
            last_fatal: None,
        }
    }

    pub fn chunk_bytes(mut self, chunk_bytes: u64) -> Self {
        self.next_chunk = chunk_bytes.max(1);
        self
    }

    pub fn transfer_cap(mut self, cap: u64) -> Self {
        self.transfer_cap = cap;
        self
    }

    /// Total bytes pulled over the network so far.
    pub fn transferred(&self) -> u64 {
        self.transferred
    }

    /// Full file size as advertised by `Content-Range`, once known.
    pub fn total_size(&self) -> Option<u64> {
        self.total_size
    }

    /// Prefer a transport-level failure over a generic parse error.
    pub fn take_last_fatal(&mut self) -> Option<String> {
        self.last_fatal.take()
    }

    fn fetch(&mut self, start: u64) -> Result<(), String> {
        let chunk = self.next_chunk;
        if start > 0 && self.transferred + chunk > self.transfer_cap {
            let msg = format!(
                "remote header inspection exceeded its {} MiB transfer budget",
                self.transfer_cap / (1024 * 1024)
            );
            self.last_fatal = Some(msg.clone());
            return Err(msg);
        }

        let end = start.saturating_add(chunk - 1);
        // Follow the resolve → CDN redirect chain ourselves so subsequent
        // chunks hit the signed CDN URL directly (one round-trip instead of
        // the whole chain per chunk).
        let mut attempt_url = self
            .resolved_url
            .clone()
            .unwrap_or_else(|| self.url.clone());
        let mut redirects = 0usize;
        let resp = loop {
            let response = ureq::get(&attempt_url)
                .header("Range", &format!("bytes={start}-{end}"))
                .config()
                .max_redirects(0)
                .timeout_global(Some(Duration::from_secs(REQUEST_TIMEOUT_SECS)))
                .build()
                .call()
                .map_err(|e| {
                    let msg = format!("range request bytes={start}-{end} failed: {e}");
                    self.last_fatal = Some(msg.clone());
                    msg
                })?;
            let status = response.status().as_u16();
            if !(300..400).contains(&status) {
                break response;
            }
            redirects += 1;
            if redirects > MAX_REDIRECTS {
                let msg = "too many redirects while resolving the file URL".to_string();
                self.last_fatal = Some(msg.clone());
                return Err(msg);
            }
            let location = response
                .headers()
                .get("Location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| {
                    let msg = format!("redirect {status} without a Location header");
                    self.last_fatal = Some(msg.clone());
                    msg
                })?
                .to_string();
            attempt_url = join_redirect_url(&attempt_url, &location)?;
        };
        if self.resolved_url.is_none() && redirects > 0 {
            self.resolved_url = Some(attempt_url);
        }

        let status = resp.status().as_u16();
        // A server may ignore Range and answer 200 with the full body; only
        // tolerable for the very first chunk since we stop reading early.
        if status != 206 && !(status == 200 && start == 0) {
            let msg = format!(
                "server did not honor byte ranges (status {status} for bytes={start}-{end})"
            );
            self.last_fatal = Some(msg.clone());
            return Err(msg);
        }

        if let Some(range) = resp
            .headers()
            .get("Content-Range")
            .and_then(|v| v.to_str().ok())
        {
            match parse_content_range_total(range) {
                Ok((first, total)) => {
                    if first != start {
                        let msg =
                            format!("server returned wrong range start ({first}, asked {start})");
                        self.last_fatal = Some(msg.clone());
                        return Err(msg);
                    }
                    self.total_size = Some(total);
                }
                Err(e) => {
                    // Non-fatal: keep going without a known total size.
                    eprintln!("warning: unparseable Content-Range '{range}': {e}");
                }
            }
        }

        let mut body = resp.into_body().into_reader();
        let want = self
            .next_chunk
            .min(self.transfer_cap.saturating_sub(self.transferred));
        let mut buf = vec![0u8; want as usize];
        let mut filled = 0usize;
        while filled < buf.len() {
            match body.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) => {
                    let msg = format!("range body read failed: {e}");
                    self.last_fatal = Some(msg.clone());
                    return Err(msg);
                }
            }
        }
        self.transferred += filled as u64;
        buf.truncate(filled);
        self.buf = buf;
        self.buf_start = Some(start);
        if filled as u64 == want {
            self.next_chunk = (self.next_chunk.saturating_mul(2)).min(MAX_CHUNK_BYTES);
        }
        Ok(())
    }

    fn ensure_window(&mut self) -> Result<(), String> {
        let covered = matches!((self.buf_start.as_ref(), self.buf.len()),
            (Some(start), len) if self.pos >= *start && (self.pos - *start) < len.max(1) as u64 && (self.pos - *start) < self.buf.len() as u64);
        if !covered {
            self.fetch(self.pos)?;
        }
        Ok(())
    }
}

impl Read for RangeReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        self.ensure_window().map_err(io::Error::other)?;
        let start = self.buf_start.expect("ensure_window filled buffer");
        let offset = (self.pos - start) as usize;
        let available = self.buf.len() - offset;
        let n = available.min(out.len());
        if n == 0 {
            // Server returned an empty body at this offset: treat as EOF.
            return Ok(0);
        }
        out[..n].copy_from_slice(&self.buf[offset..offset + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for RangeReader {
    fn seek(&mut self, whence: SeekFrom) -> io::Result<u64> {
        let target = match whence {
            SeekFrom::Start(offset) => offset,
            SeekFrom::Current(delta) => i128::from(self.pos)
                .checked_add(i128::from(delta))
                .filter(|v| *v >= 0 && *v <= u64::MAX as i128)
                .ok_or_else(|| io::Error::other("seek out of bounds"))?
                as u64,
            SeekFrom::End(delta) => {
                let total = self.total_size.ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::Unsupported,
                        "SeekFrom::End needs a known total size (fetch first)",
                    )
                })?;
                i128::from(total)
                    .checked_add(i128::from(delta))
                    .filter(|v| *v >= 0)
                    .ok_or_else(|| io::Error::other("seek before start of file"))?
                    as u64
            }
        };
        self.pos = target;
        Ok(target)
    }
}

/// Resolve a `Location` header against the URL it came from. Handles the
/// absolute URLs CDNs issue; relative paths are joined at the root.
fn join_redirect_url(base: &str, location: &str) -> Result<String, String> {
    let location = location.trim();
    if location.is_empty() {
        return Err("empty redirect Location".to_string());
    }
    if location.starts_with("http://") || location.starts_with("https://") {
        return Ok(location.to_string());
    }
    let scheme_end = base
        .find("://")
        .ok_or_else(|| format!("cannot resolve relative redirect '{location}' against '{base}'"))?;
    let after_scheme = &base[scheme_end + 3..];
    let authority_end = after_scheme.find('/').unwrap_or(after_scheme.len());
    let authority = &after_scheme[..authority_end];
    if location.starts_with('/') {
        let scheme = &base[..scheme_end];
        return Ok(format!("{scheme}://{authority}{location}"));
    }
    Err(format!(
        "unsupported relative redirect '{location}' (expected absolute or root-relative)"
    ))
}

/// Parse `"bytes first-last/total"` from a Content-Range header value.
fn parse_content_range_total(value: &str) -> Result<(u64, u64), String> {
    let rest = value
        .trim()
        .strip_prefix("bytes ")
        .ok_or_else(|| "missing 'bytes ' unit".to_string())?;
    let (range_part, total_part) = rest
        .split_once('/')
        .ok_or_else(|| "missing '/' separator".to_string())?;
    let first = range_part
        .split_once('-')
        .ok_or_else(|| "missing '-' in range".to_string())?
        .0
        .trim()
        .parse::<u64>()
        .map_err(|e| format!("bad range start: {e}"))?;
    let total = if total_part.trim() == "*" {
        return Err("unknown total ('*')".to_string());
    } else {
        total_part
            .trim()
            .parse::<u64>()
            .map_err(|e| format!("bad total size: {e}"))?
    };
    Ok((first, total))
}

/// A GGUF header parsed from a remote file, plus transfer accounting.
pub struct RemoteGguf {
    pub header: GgufHeader,
    /// Advertised full size of the remote file (from Content-Range).
    pub file_size_bytes: Option<u64>,
    /// Bytes actually pulled over the network to build the summary.
    pub transferred_bytes: u64,
}

impl RemoteGguf {
    /// Summarize the parsed header (same shape as the local audit).
    pub fn summarize(&self) -> GgufModelSummary {
        GgufModelSummary::from_header(&self.header)
    }
}

/// Fetch and parse a GGUF header from a direct URL using range requests.
pub fn open_remote_gguf(url: &str) -> Result<RemoteGguf, String> {
    let mut reader = RangeReader::new(url);
    let parsed = GgufHeader::read_from(&mut reader);
    match parsed {
        Ok(header) => Ok(RemoteGguf {
            header,
            file_size_bytes: reader.total_size(),
            transferred_bytes: reader.transferred(),
        }),
        Err(parse_error) => Err(reader.take_last_fatal().unwrap_or(parse_error)),
    }
}

/// Resolve `{owner}/{repo}` (+ optional quant variant like `Q4_K_M`) to a
/// concrete GGUF resolve URL via the HF tree API. Prefers the requested
/// variant; otherwise reuses the catalog's quality preference order without
/// a memory budget (auditing is introspection, not a recommendation).
pub fn resolve_repo_gguf_url(
    repo_id: &str,
    quant_variant: Option<&str>,
) -> Result<(String, String, u64), String> {
    use crate::providers::LlamaCppProvider;

    let files = LlamaCppProvider::list_repo_gguf_files(repo_id);
    if files.is_empty() {
        return Err(format!("no GGUF files found in repo '{repo_id}'"));
    }

    if let Some(variant) = quant_variant {
        let needle = variant.to_ascii_uppercase();
        let mut matches: Vec<&(String, u64)> = files
            .iter()
            .filter(|(name, _)| name.to_ascii_uppercase().contains(&needle))
            .collect();
        if matches.is_empty() {
            let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
            return Err(format!(
                "no GGUF file matching quant '{variant}' in repo '{repo_id}' (files: {})",
                names.join(", ")
            ));
        }
        // Single-file repos win over shard sets; within shards take shard 1.
        matches.sort_by_key(|(name, _)| (name.contains("-of-"), name.clone()));
        let (filename, size) = matches[0];
        return Ok((filename.clone(), resolve_url(repo_id, filename), *size));
    }

    let (filename, size) = LlamaCppProvider::select_best_gguf(&files, f64::INFINITY)
        .ok_or_else(|| format!("no usable GGUF candidate in repo '{repo_id}'"))?;
    Ok((filename.clone(), resolve_url(repo_id, &filename), size))
}

fn resolve_url(repo_id: &str, filename: &str) -> String {
    format!("https://huggingface.co/{repo_id}/resolve/main/{filename}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Serve one connection: either a 206 slice of `blob` or, when
    /// `ignore_ranges`, a plain 200 with the whole body.
    fn answer_range_request(stream: &mut std::net::TcpStream, blob: &[u8], ignore_ranges: bool) {
        use std::io::Write as _;

        let request = read_http_request_head(stream);
        let requested = request.as_deref().and_then(|text| {
            text.lines().find_map(|l| {
                let (name, value) = l.split_once(':')?;
                name.eq_ignore_ascii_case("range")
                    .then(|| value.trim().strip_prefix("bytes=").map(str::to_string))
                    .flatten()
            })
        });

        let range = requested.filter(|_| !ignore_ranges);
        let (status, extra, payload): (&str, String, &[u8]) = match range {
            None => ("200 OK", String::new(), blob),
            Some(raw) => {
                let (a, b) = raw.split_once('-').expect("range dash");
                let start: usize = a.parse().expect("range start");
                let end: usize = b.parse().expect("range end");
                let end = end.min(blob.len().saturating_sub(1).max(start));
                (
                    "206 Partial Content",
                    format!("Content-Range: bytes {start}-{end}/{}\r\n", blob.len()),
                    &blob[start..=end],
                )
            }
        };
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
            payload.len()
        );
        stream.write_all(response.as_bytes()).ok();
        stream.write_all(payload).ok();
    }

    /// Read bytes until the blank line terminating the HTTP request head.
    fn read_http_request_head(stream: &mut std::net::TcpStream) -> Option<String> {
        use std::io::Read as _;

        let mut request = Vec::new();
        let mut byte = [0u8; 1];
        while !request.ends_with(b"\r\n\r\n") {
            if stream.read(&mut byte).unwrap_or(0) == 0 {
                return None;
            }
            request.push(byte[0]);
            if request.len() > 16 * 1024 {
                return None;
            }
        }
        Some(String::from_utf8_lossy(&request).into_owned())
    }

    /// Consume and discard an incoming HTTP request head.
    fn drain_request(stream: &mut std::net::TcpStream) {
        read_http_request_head(stream);
    }

    /// Accept loop answering every connection with range slices of `blob`.
    fn serve_ranges(listener: TcpListener, blob: Vec<u8>, hits: Arc<AtomicUsize>) {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { break };
            let blob = Arc::new(blob.clone());
            let hits = Arc::clone(&hits);
            std::thread::spawn(move || {
                hits.fetch_add(1, Ordering::SeqCst);
                answer_range_request(&mut stream, &blob, false);
            });
        }
    }

    /// Minimal range-serving TCP server over [`serve_ranges`].
    fn spawn_range_server(blob: Vec<u8>, ignore_ranges: bool) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_clone = Arc::clone(&hits);
        let blob = Arc::new(blob);
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut stream) = conn else { break };
                let hits = Arc::clone(&hits_clone);
                let blob = Arc::clone(&blob);
                std::thread::spawn(move || {
                    hits.fetch_add(1, Ordering::SeqCst);
                    answer_range_request(&mut stream, &blob, ignore_ranges);
                });
            }
        });
        (format!("http://{addr}/model.gguf"), hits)
    }

    fn sample_blob(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn read_all(reader: &mut RangeReader, len: usize) -> Vec<u8> {
        let mut out = Vec::new();
        reader
            .by_ref()
            .take(len as u64)
            .read_to_end(&mut out)
            .unwrap();
        out
    }

    #[test]
    fn sequential_read_spans_multiple_requests_with_growing_chunks() {
        let blob = sample_blob(700_000); // 512 KiB first chunk + remainder
        let (url, hits) = spawn_range_server(blob.clone(), false);
        let mut rr = RangeReader::new(url).chunk_bytes(256 * 1024);
        assert_eq!(read_all(&mut rr, blob.len()), blob);
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        assert_eq!(rr.transferred(), blob.len() as u64);
        assert_eq!(rr.total_size(), Some(blob.len() as u64));
    }

    #[test]
    fn seek_backward_refetches_correct_window() {
        let blob = sample_blob(300_000);
        let (url, _) = spawn_range_server(blob.clone(), false);
        let mut rr = RangeReader::new(url);
        let mut tail = [0u8; 8];
        rr.seek(SeekFrom::Start(250_000)).unwrap();
        rr.read_exact(&mut tail).unwrap();
        assert_eq!(&tail, &blob[250_000..250_008]);

        rr.seek(SeekFrom::Start(42)).unwrap();
        let mut head = [0u8; 16];
        rr.read_exact(&mut head).unwrap();
        assert_eq!(&head, &blob[42..58]);
    }

    #[test]
    fn seek_current_skips_without_reading_through() {
        let blob = sample_blob(100_000);
        let (url, hits) = spawn_range_server(blob.clone(), false);
        let mut rr = RangeReader::new(url).chunk_bytes(64 * 1024);
        rr.seek(SeekFrom::Start(0)).unwrap();
        let mut first = [0u8; 4];
        rr.read_exact(&mut first).unwrap();

        rr.seek(SeekFrom::Current(90_000)).unwrap(); // beyond current window
        let mut late = [0u8; 4];
        rr.read_exact(&mut late).unwrap();
        assert_eq!(&late, &blob[90004..90008]);
        // First chunk (64 KiB) + second fetch starting at 90004 — not ~90 KB
        // of streamed-through scratch reads.
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        assert!(rr.transferred() < 140_000);
    }

    #[test]
    fn transfer_budget_is_enforced() {
        let blob = sample_blob(600_000);
        let (url, _) = spawn_range_server(blob, false);
        let mut rr = RangeReader::new(url)
            .chunk_bytes(64 * 1024)
            .transfer_cap(100_000);
        rr.seek(SeekFrom::Start(50_000)).unwrap(); // forces a second fetch
        let err = rr
            .by_ref()
            .take(500_000)
            .read_to_end(&mut Vec::new())
            .unwrap_err();
        assert!(err.to_string().contains("transfer budget"), "{err}");
        assert!(rr.take_last_fatal().is_some());
    }

    #[test]
    fn server_without_range_support_errors_clearly_after_first_chunk() {
        let blob = sample_blob(400_000);
        let (url, _) = spawn_range_server(blob.clone(), true);
        let mut rr = RangeReader::new(url).chunk_bytes(64 * 1024);
        let mut head = [0u8; 4];
        rr.read_exact(&mut head).unwrap(); // 200 tolerated at offset 0
        assert_eq!(&head, &blob[..4]);

        rr.seek(SeekFrom::Start(200_000)).unwrap();
        let err = rr
            .by_ref()
            .take(1000)
            .read_to_end(&mut Vec::new())
            .unwrap_err();
        assert!(err.to_string().contains("byte ranges"), "{err}");
    }

    #[test]
    fn content_range_parser_handles_known_shapes() {
        assert_eq!(
            parse_content_range_total("bytes 0-262143/485452288"),
            Ok((0, 485452288))
        );
        assert_eq!(
            parse_content_range_total("bytes 512-767/*"),
            Err("unknown total ('*')".to_string())
        );
        assert!(parse_content_range_total("items 0-1/2").is_err());
    }

    #[test]
    fn parses_remote_gguf_header_over_ranges() {
        // Hand-built minimal v3 header whose metadata includes an array big
        // enough to force skips across several range requests.
        let kvs: Vec<Vec<u8>> = vec![
            str_kv("general.architecture", "llama"),
            u32_kv("llama.block_count", 2),
            u32_kv("llama.attention.head_count", 8),
        ];
        let filler: Vec<u8> = (0..150_000u32).flat_map(|i| i.to_le_bytes()).collect();
        let mut array_kv = lp_string("tokenizer.ggml.scores");
        array_kv.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        array_kv.extend_from_slice(&6u32.to_le_bytes()); // of F32 elements
        array_kv.extend_from_slice(&(filler.len() as u64 / 4).to_le_bytes());
        array_kv.extend_from_slice(&filler);

        let tensor = {
            let name = b"token_embd.weight";
            let mut b = (name.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(name);
            b.extend_from_slice(&2u32.to_le_bytes());
            b.extend_from_slice(&256u64.to_le_bytes());
            b.extend_from_slice(&128u64.to_le_bytes());
            b.extend_from_slice(&1u32.to_le_bytes()); // F16
            b.extend_from_slice(&0u64.to_le_bytes());
            b
        };

        let mut blob = b"GGUF".to_vec();
        blob.extend_from_slice(&3u32.to_le_bytes());
        blob.extend_from_slice(&1u64.to_le_bytes()); // tensors
        blob.extend_from_slice(&4u64.to_le_bytes()); // kvs
        for kv in &kvs {
            blob.extend_from_slice(kv);
        }
        blob.extend_from_slice(&array_kv);
        blob.extend_from_slice(&tensor);

        let (url, hits) = spawn_range_server(blob.clone(), false);
        let mut rr = RangeReader::new(url).chunk_bytes(32 * 1024);
        let header = match GgufHeader::read_from(&mut rr) {
            Ok(h) => h,
            Err(e) => panic!(
                "parse failed: {e}; transferred={} hits={}",
                rr.transferred(),
                hits.load(Ordering::SeqCst)
            ),
        };
        eprintln!(
            "DEBUG transferred={} total={:?} hits={}",
            rr.transferred(),
            rr.total_size(),
            hits.load(Ordering::SeqCst)
        );

        assert_eq!(header.get_str("general.architecture"), Some("llama"));
        assert_eq!(header.get_u64("llama.block_count"), Some(2));
        assert_eq!(header.tensors.len(), 1);
        assert_eq!(header.tensors[0].name, "token_embd.weight");
        // The 150 KB array was skipped, not downloaded.
        assert!(rr.transferred() < blob.len() as u64 / 2);
    }

    fn lp_string(s: &str) -> Vec<u8> {
        let mut b = (s.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(s.as_bytes());
        b
    }

    fn str_kv(key: &str, value: &str) -> Vec<u8> {
        let mut b = lp_string(key);
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&lp_string(value));
        b
    }

    fn u32_kv(key: &str, value: u32) -> Vec<u8> {
        let mut b = lp_string(key);
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&value.to_le_bytes());
        b
    }

    #[test]
    fn redirect_is_resolved_once_then_reused_directly() {
        // The origin answers every request with a 302 pointing at the CDN;
        // only the FIRST chunk may pass through it. Afterwards all requests
        // must hit the CDN URL directly.
        let blob = sample_blob(300_000);
        let cdn = TcpListener::bind("127.0.0.1:0").expect("bind cdn");
        let cdn_addr = cdn.local_addr().unwrap();
        let hits_cdn = Arc::new(AtomicUsize::new(0));
        let blob_for_cdn = blob.clone();
        let hits_for_cdn = Arc::clone(&hits_cdn);
        std::thread::spawn(move || serve_ranges(cdn, blob_for_cdn, hits_for_cdn));

        let origin = TcpListener::bind("127.0.0.1:0").expect("bind origin");
        let origin_url = format!("http://{}/origin/model.gguf", origin.local_addr().unwrap());
        let hits_origin = Arc::new(AtomicUsize::new(0));
        let location = format!("http://{cdn_addr}/weights.bin");
        let h = Arc::clone(&hits_origin);
        std::thread::spawn(move || {
            for conn in origin.incoming().flatten() {
                let mut stream = conn;
                drain_request(&mut stream);
                h.fetch_add(1, Ordering::SeqCst);
                use std::io::Write as _;
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                        .as_bytes(),
                    )
                    .ok();
            }
        });

        let mut rr = RangeReader::new(origin_url).chunk_bytes(64 * 1024);
        assert_eq!(read_all(&mut rr, blob.len()), blob);
        assert_eq!(hits_origin.load(Ordering::SeqCst), 1);
        assert!(hits_cdn.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn seek_end_requires_known_total() {
        let blob = sample_blob(10_000);
        let (url, _) = spawn_range_server(blob, false);
        let mut rr = RangeReader::new(url);
        assert!(rr.seek(SeekFrom::End(-4)).is_err());
    }
}

#[cfg(test)]
mod network_tests {
    use super::*;

    fn network_enabled() -> bool {
        std::env::var("LLMFIT_NET_TESTS").is_ok()
    }

    #[test]
    #[ignore = "requires network access to huggingface.co (run with LLMFIT_NET_TESTS=1)"]
    fn criterion_big_moe_header_within_transfer_cap() {
        if !network_enabled() {
            eprintln!("skipping: LLMFIT_NET_TESTS not set");
            return;
        }
        let started = std::time::Instant::now();
        let (filename, url, _size) =
            resolve_repo_gguf_url("Qwen/Qwen3-235B-A22B-GGUF", Some("Q8_0")).expect("resolve");
        println!("selected {filename}");
        let remote = open_remote_gguf(&url).expect("remote open");
        let elapsed = started.elapsed();

        assert_eq!(
            remote.header.get_str("general.architecture"),
            Some("qwen3moe")
        );
        assert_eq!(remote.header.get_u64("qwen3moe.expert_count"), Some(128));
        assert_eq!(remote.header.get_u64("qwen3moe.expert_used_count"), Some(8));
        assert!(remote.transferred_bytes <= DEFAULT_TRANSFER_CAP_BYTES);
        assert!(
            elapsed.as_secs() < 30,
            "header inspection took too long: {elapsed:?}"
        );
        println!(
            "transferred {} KiB in {elapsed:?} (file advertised {:?})",
            remote.transferred_bytes / 1024,
            remote.file_size_bytes
        );
    }

    #[test]
    #[ignore = "requires network access to huggingface.co (run with LLMFIT_NET_TESTS=1)"]
    fn small_dense_repo_audits_cleanly() {
        if !network_enabled() {
            eprintln!("skipping: LLMFIT_NET_TESTS not set");
            return;
        }
        let (_filename, url, _size) =
            resolve_repo_gguf_url("bartowski/Llama-3.2-1B-Instruct-GGUF", Some("Q4_K_M"))
                .expect("resolve");
        let remote = open_remote_gguf(&url).expect("remote open");
        assert_eq!(remote.header.get_str("general.architecture"), Some("llama"));
        assert_eq!(remote.header.get_u64("llama.block_count"), Some(16));
    }
}
