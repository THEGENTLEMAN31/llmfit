use assert_cmd::Command;
use serde_json::Value;

fn run_json_command(args: &[&str]) -> Value {
    let output = Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .args(args)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    serde_json::from_slice(&output).expect("command did not emit valid JSON")
}

fn models_array(json: &Value) -> &[Value] {
    json.get("models")
        .and_then(Value::as_array)
        .expect("JSON output missing models array")
}

#[test]
fn help_includes_project_description() {
    let output = Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .arg("--help")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let text = String::from_utf8(output).expect("--help output was not UTF-8");
    assert!(text.contains("Right-size LLM models to your system's hardware"));
}

#[test]
fn version_matches_package_version() {
    let output = Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .arg("--version")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let text = String::from_utf8(output).expect("--version output was not UTF-8");
    assert!(text.contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn system_json_has_expected_shape() {
    let json = run_json_command(&["--no-dashboard", "--json", "system"]);
    let system = json
        .get("system")
        .and_then(Value::as_object)
        .expect("system key missing or not an object");

    assert!(system.contains_key("available_ram_gb"));
    assert!(system.contains_key("cpu_cores"));
    assert!(system.contains_key("backend"));
}

#[test]
fn list_json_returns_non_empty_catalog() {
    let json = run_json_command(&["--no-dashboard", "--json", "list"]);
    let models = json
        .as_array()
        .expect("list --json output should be an array");

    assert!(!models.is_empty(), "model catalog should not be empty");
    let first = models[0]
        .as_object()
        .expect("first model entry should be a JSON object");
    assert!(first.contains_key("name"));
    assert!(first.contains_key("provider"));
}

#[test]
fn fit_json_obeys_limit_and_contains_models_field() {
    let json = run_json_command(&[
        "--no-dashboard",
        "--json",
        "--memory",
        "8G",
        "--ram",
        "16G",
        "--cpu-cores",
        "4",
        "fit",
        "--limit",
        "3",
    ]);

    let models = json
        .get("models")
        .and_then(Value::as_array)
        .expect("fit --json output missing models array");

    assert!(models.len() <= 3, "fit output exceeded requested limit");

    if let Some(first) = models.first() {
        let first = first
            .as_object()
            .expect("fit model entry should be a JSON object");
        assert!(first.contains_key("fit_level"));
        assert!(first.contains_key("run_mode"));
        assert!(first.contains_key("score"));
    }
}

#[test]
fn fit_provider_filter_accepts_commas_case_insensitively_and_matches_gguf_sources() {
    let json = run_json_command(&[
        "--no-dashboard",
        "--json",
        "--memory",
        "8G",
        "--ram",
        "16G",
        "--cpu-cores",
        "4",
        "fit",
        "--providers",
        "not-a-provider,BaRtOwSkI",
        "--limit",
        "3",
    ]);
    let models = models_array(&json);

    assert!(
        !models.is_empty(),
        "GGUF-source provider should match models"
    );
    assert!(
        models.len() <= 3,
        "provider filter should apply before the limit"
    );
    assert!(models.iter().all(|model| {
        model
            .get("provider")
            .and_then(Value::as_str)
            .is_some_and(|provider| !provider.eq_ignore_ascii_case("bartowski"))
    }));
}

#[test]
fn recommend_capability_filter_does_not_ignore_unknown_or_tts() {
    let tts_json = run_json_command(&[
        "--no-dashboard",
        "--json",
        "--memory",
        "8G",
        "--ram",
        "16G",
        "--cpu-cores",
        "4",
        "recommend",
        "--capability",
        "tts",
        "-n",
        "5",
    ]);
    assert!(models_array(&tts_json).iter().all(|model| {
        model
            .get("capability_ids")
            .and_then(Value::as_array)
            .is_some_and(|caps| caps.iter().any(|cap| cap.as_str() == Some("tts")))
    }));

    let unknown_json = run_json_command(&[
        "--no-dashboard",
        "--json",
        "--memory",
        "8G",
        "--ram",
        "16G",
        "--cpu-cores",
        "4",
        "recommend",
        "--capability",
        "not_a_capability",
        "-n",
        "5",
    ]);
    assert!(
        models_array(&unknown_json).is_empty(),
        "unknown capability should not match every model"
    );
}

#[test]
fn fit_json_returns_empty_models_when_no_perfect_matches() {
    let json = run_json_command(&[
        "--no-dashboard",
        "--json",
        "--memory",
        "1M",
        "--ram",
        "1M",
        "--cpu-cores",
        "1",
        "fit",
        "--perfect",
    ]);

    let models = json
        .get("models")
        .and_then(Value::as_array)
        .expect("fit --json output missing models array");

    assert!(
        models.is_empty(),
        "expected no perfect matches on extremely constrained hardware"
    );
}

#[test]
fn cpu_cores_parser_rejects_zero() {
    Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .args(["--cpu-cores", "0", "--json", "system"])
        .assert()
        .failure();
}

// ─── gguf audit ─────────────────────────────────────────────────────────────

/// Hand-assembled minimal GGUF v3 header (llama arch, mixed quant tensors).
fn write_gguf_fixture(path: &std::path::Path) {
    use std::io::Write;

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());

    let kvs: Vec<(String, u32, Vec<u8>)> = vec![
        ("general.architecture".into(), 8, {
            let v = b"llama";
            let mut b = (v.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(v);
            b
        }),
        ("llama.block_count".into(), 4, 2u32.to_le_bytes().to_vec()),
        (
            "llama.attention.head_count".into(),
            4,
            8u32.to_le_bytes().to_vec(),
        ),
        (
            "llama.attention.head_count_kv".into(),
            4,
            4u32.to_le_bytes().to_vec(),
        ),
        (
            "llama.context_length".into(),
            4,
            4096u32.to_le_bytes().to_vec(),
        ),
    ];

    let tensor = |name: &str, dims: &[u64], ty: u32| -> Vec<u8> {
        let mut b = (name.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b
    };

    let tensors = vec![
        tensor("token_embd.weight", &[256, 128], 1),    // F16
        tensor("blk.0.attn_q.weight", &[128, 128], 12), // Q4_K
        tensor("blk.0.ffn_down.weight", &[128, 512], 12),
        tensor("blk.1.attn_q.weight", &[128, 128], 12),
        tensor("blk.1.ffn_down.weight", &[128, 512], 12),
    ];

    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (_key, _ty, payload) in &kvs {
        // key string then value type id then payload
        let key = &_key; // silence unused warnings in tuple destructure above
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(&_ty.to_le_bytes());
        out.extend_from_slice(payload);
    }
    for t in &tensors {
        out.extend_from_slice(t);
    }

    let mut f = std::fs::File::create(path).expect("create fixture");
    f.write_all(&out).expect("write fixture");
}

#[test]
fn audit_json_reports_real_header_data() {
    let dir = std::env::temp_dir().join(format!("llmfit-audit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("fixture-q4_k_m.gguf");
    write_gguf_fixture(&path);

    let json = run_json_command(&[
        "--no-dashboard",
        "--json",
        "audit",
        path.to_str().expect("utf8 path"),
    ]);

    assert_eq!(json["summary"]["architecture"], "llama");
    assert_eq!(json["summary"]["block_count"], 2);
    assert_eq!(json["summary"]["attention_heads"], 8);
    assert_eq!(json["summary"]["key_value_heads"], 4);
    assert_eq!(json["summary"]["context_length"], 4096);
    assert_eq!(json["summary"]["dominant_quant_label"], "Q4_K");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn audit_rejects_non_gguf_file_with_error() {
    let dir = std::env::temp_dir().join(format!("llmfit-audit-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("not-a-model.gguf");
    std::fs::write(&path, b"definitely not a gguf header").expect("write junk");

    Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .args(["--no-dashboard", "audit", path.to_str().expect("utf8 path")])
        .assert()
        .failure();

    std::fs::remove_dir_all(&dir).ok();
}

/// Minimal MoE-shaped GGUF: 4 blocks, each with routed-expert tensors, so
/// `plan` has real expert mass to derive a --n-cpu-moe split from.
fn write_moe_gguf_fixture(path: &std::path::Path) {
    use std::io::Write;

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());

    let str_val = |s: &str| -> Vec<u8> {
        let mut b = (s.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(s.as_bytes());
        b
    };
    let u32_val = |v: u32| -> Vec<u8> { v.to_le_bytes().to_vec() };

    let kvs: Vec<(String, u32, Vec<u8>)> = vec![
        ("general.architecture".into(), 8, str_val("llama")),
        ("llama.block_count".into(), 4, u32_val(4)),
        ("llama.attention.head_count".into(), 4, u32_val(16)),
        ("llama.attention.head_count_kv".into(), 4, u32_val(4)),
        ("llama.attention.key_length".into(), 4, u32_val(128)),
        ("llama.context_length".into(), 4, u32_val(40960)),
        ("llama.expert_count".into(), 4, u32_val(8)),
        ("llama.expert_used_count".into(), 4, u32_val(2)),
    ];

    let tensor = |name: &str, dims: &[u64], ty: u32| -> Vec<u8> {
        let mut b = (name.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b
    };

    let mut tensors = vec![tensor("token_embd.weight", &[256, 128], 1)];
    for blk in 0..4u64 {
        tensors.push(tensor(&format!("blk.{blk}.attn_q.weight"), &[128, 128], 12));
        // Routed experts: [ffn_dim, hidden, n_expert] — the mass llama.cpp
        // moves with --n-cpu-moe. Sized (~30 GB at Q4_K) so it cannot fit
        // whole into the fixture machine's 24 GB of VRAM.
        tensors.push(tensor(
            &format!("blk.{blk}.ffn_gate_exps.weight"),
            &[16_000_000, 128, 8],
            12,
        ));
    }

    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (_key, _ty, payload) in &kvs {
        out.extend_from_slice(&(_key.len() as u64).to_le_bytes());
        out.extend_from_slice(_key.as_bytes());
        out.extend_from_slice(&_ty.to_le_bytes());
        out.extend_from_slice(payload);
    }
    for t in &tensors {
        out.extend_from_slice(t);
    }

    let mut f = std::fs::File::create(path).expect("create moe fixture");
    f.write_all(&out).expect("write moe fixture");
}

#[test]
fn plan_local_moe_gguf_prints_llamacpp_command_on_fixture_hardware() {
    let dir = std::env::temp_dir().join(format!("llmfit-plan-moe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("fixture-moe-q4_k.gguf");
    write_moe_gguf_fixture(&path);

    // Fixture machine from FORK_GUIDE §1: RTX 3090-class VRAM + big DDR5.
    let output = Command::cargo_bin("llmfit")
        .expect("failed to locate llmfit test binary")
        .args([
            "--no-dashboard",
            "--memory",
            "24G",
            "--ram",
            "96G",
            "--cpu-cores",
            "16",
            "plan",
            path.to_str().expect("utf8 path"),
            "--context",
            "16384",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).expect("utf8 stdout");

    assert!(
        stdout.contains("Suggested llama.cpp command:"),
        "plan must surface an actionable command:\n{stdout}"
    );
    assert!(stdout.contains("llama-server"), "{stdout}");
    assert!(stdout.contains("-m "), "{stdout}");
    assert!(stdout.contains("-c 16384"), "{stdout}");
    assert!(stdout.contains("-fa"), "{stdout}");
    // The fixture carries routed-expert tensors and the machine is
    // GPU-constrained, so the command must contain a concrete MoE split.
    assert!(
        stdout.contains("--n-cpu-moe "),
        "expected a concrete --n-cpu-moe split on constrained VRAM:\n{stdout}"
    );

    // JSON consumers get the same command as a first-class field.
    let json = run_json_command(&[
        "--no-dashboard",
        "--json",
        "--memory",
        "24G",
        "--ram",
        "96G",
        "plan",
        path.to_str().unwrap(),
        "--context",
        "16384",
    ]);
    let cmd = json["llamacpp_command"]
        .as_str()
        .expect("JSON plan must carry llamacpp_command");
    assert!(cmd.contains("--n-cpu-moe "), "{cmd}");

    std::fs::remove_dir_all(&dir).ok();
}
