"""python -m novex.site [--port 8080] [--host 127.0.0.1] [--data site_data]"""
import argparse

import uvicorn

from .app import create_app


def main():
    ap = argparse.ArgumentParser(description="Novex detector website")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--data", default="site_data", help="where uploads and results are kept")
    a = ap.parse_args()
    print(f"Novex detector: http://{a.host if a.host != '0.0.0.0' else 'localhost'}:{a.port}")
    uvicorn.run(create_app(a.data), host=a.host, port=a.port, log_level="warning")


main()
