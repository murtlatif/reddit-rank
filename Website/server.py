import cgi
import cgitb
from http.server import CGIHTTPRequestHandler, HTTPServer

class Handler(CGIHTTPRequestHandler):
    cgi_directories = ["/cgi-bin"]

PORT = 8080

def main():
    httpd = HTTPServer(("", PORT), Handler)
    print('Serving at port', PORT)
    httpd.serve_forever()

if __name__ == '__main__':
    main()