FROM nginx:1.27-alpine

COPY nginx.conf /tmp/nginx.conf
RUN tail -c +4 /tmp/nginx.conf > /etc/nginx/nginx.conf \
    && nginx -t \
    && rm /tmp/nginx.conf
COPY index.html /usr/share/nginx/html/index.html
