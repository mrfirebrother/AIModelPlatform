from pathlib import Path


def test_frontend_dockerfile_strips_nginx_config_bom_and_validates() -> None:
    dockerfile = Path(__file__).parents[3] / "frontend" / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8-sig")

    assert "COPY nginx.conf /tmp/nginx.conf" in content
    assert "tail -c +4 /tmp/nginx.conf > /etc/nginx/nginx.conf" in content
    assert "nginx -t" in content
    assert "COPY nginx.conf /etc/nginx/nginx.conf" not in content
