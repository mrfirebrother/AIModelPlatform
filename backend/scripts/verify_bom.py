from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET_DIRECTORIES = (ROOT / "backend", ROOT / "frontend", ROOT / "infra")
TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yml",
    ".yaml",
    ".example",
}
TEXT_FILENAMES = {"Dockerfile"}
BOM = b"\xef\xbb\xbf"


def is_target_text_file(path: Path) -> bool:
    return path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_SUFFIXES


SKIP_DIRS = {"node_modules", "__pycache__", ".pytest_cache", "test-results", "dist"}


def main() -> int:
    missing_bom: list[Path] = []
    for directory in TARGET_DIRECTORIES:
        for path in directory.rglob("*"):
            if path.is_file() and is_target_text_file(path):
                if any(skip in path.parts for skip in SKIP_DIRS):
                    continue
                if path.read_bytes()[:3] != BOM:
                    missing_bom.append(path)

    if missing_bom:
        print("Missing UTF-8 BOM:")
        for path in missing_bom:
            print(path.relative_to(ROOT))
        return 1

    print("BOM check passed for backend/frontend/infra text files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
