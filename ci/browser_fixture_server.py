"""Static browser fixture server with portable JavaScript module MIME types."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript"}

    def log_message(self, _format: str, *_args: object) -> None:
        # Hundreds of module requests can fill a captured web-server pipe and
        # stall the fixture during a long browser run.
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
