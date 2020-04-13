import socket
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer

import renderer.pipeline.json as js
from renderer.pipeline.json import SetValueMessage


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class BasicHandler(BaseHTTPRequestHandler):

    # Handler for GET and set requests

    @staticmethod
    def get_behaviours(rest_url):
        message = js.node_behaviours_to_json()
        return message

    def get_graph(self, rest_url):
        message = js.graph_to_json(self.server.generator)
        return message

    def set_node(self, json_message, rest_url):
        message = SetValueMessage.parse_raw(json_message)
        for k, v in message.values.items():
            self.server.generator[message.id].bind(k, v)

    def activate_preset(self, rest_url):
        return js.PresetMessage(
            values=self.server.generator.set_from_preset(str(rest_url[0]))
        )

    @staticmethod
    def _split_url(path):
        splitpath = path.split("/")
        operation_name = splitpath[1][1:]
        rest_url = splitpath[2:]
        return operation_name, rest_url

    def _set_headers(self):
        # triggers a log message
        self.send_response(200)
        self.send_header("Content-type", "json")
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        operation_name, rest_url = self._split_url(self.path)
        message = getattr(self, operation_name)(rest_url)
        json_message = message.json()
        encoded_message = json_message.encode("utf-8")
        self.wfile.write(encoded_message)

    def do_POST(self):
        self._set_headers()
        content_length = int(self.headers["Content-Length"])
        json_data = self.rfile.read(content_length)
        operation_name, rest_url = self._split_url(self.path)
        getattr(self, operation_name)(json_data, rest_url)


def start_server(generator, server_address=("localhost", 50036)):
    httpd = HTTPServer(server_address, BasicHandler)
    httpd.generator = generator
    print("BPM Server running ...")
    httpd.serve_forever()
