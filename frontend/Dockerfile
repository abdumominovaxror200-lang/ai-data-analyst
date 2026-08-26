# syntax=docker/dockerfile:1
#
# Multi-stage build for the React/Vite frontend.
#
#   build — installs deps with `npm ci` and runs the project's real build
#           script (`tsc -b && vite build`, see package.json), producing
#           static assets in dist/.
#   serve — nginx:alpine serving those static assets, with /api/* proxied to
#           the backend container by its docker-compose service name.
#
# VITE_API_BASE_URL is a Vite *build-time* env var (frontend/src/api/client.ts
# reads it via import.meta.env at build time, defaulting to
# http://localhost:8000/api for local `npm run dev`). For the containerized
# setup we build with VITE_API_BASE_URL=/api — a same-origin relative path —
# so the browser calls nginx, not the backend directly, and nginx's own
# location /api/ proxy_pass (nginx.conf below) forwards those requests to the
# `backend` service. This keeps the build-time base URL and the runtime proxy
# consistent and avoids CORS entirely for the dockerized deployment.

FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build


FROM nginx:alpine AS serve

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
