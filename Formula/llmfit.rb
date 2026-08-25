class Llmfit < Formula
  desc "Right-size LLM models to your system's hardware"
  homepage "https://github.com/THEGENTLEMAN31/llmfit"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/THEGENTLEMAN31/llmfit/releases/download/v1.1.10/llmfit-aarch64-apple-darwin.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256"
    else
      url "https://github.com/THEGENTLEMAN31/llmfit/releases/download/v1.1.10/llmfit-x86_64-apple-darwin.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/THEGENTLEMAN31/llmfit/releases/download/v1.1.10/llmfit-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256"
    end
  end

  def install
    bin.install "llmfit"
    generate_completions_from_executable(bin/"llmfit", "completion")
  end

  test do
    assert_match "llmfit", shell_output("#{bin}/llmfit --version")
    assert_match "system", shell_output("#{bin}/llmfit system")
  end
end
