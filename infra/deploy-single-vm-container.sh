#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="__DEPLOY_IMAGE__"
CONTAINER_NAME="videopoc"
ROLLBACK_CONTAINER="${CONTAINER_NAME}-previous"
ENV_FILE="/home/azureuser/videopoc/.env"
LOGIN_SERVER="${IMAGE%%/*}"
ACR_NAME="${LOGIN_SERVER%%.*}"
previous_available=0
current_stopped=0
new_container_started=0

if [[ "$IMAGE" == *DEPLOY_IMAGE* || "$IMAGE" != */*:* ]]; then
  echo "A fully qualified deployment image is required." >&2
  exit 1
fi

for command in az docker curl; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command is missing: $command" >&2
    exit 1
  fi
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file does not exist: $ENV_FILE" >&2
  exit 1
fi

for directory in \
  /home/azureuser/vp/models \
  /home/azureuser/vp/data \
  /home/azureuser/vp/gallery \
  /home/azureuser/vp/results \
  /home/azureuser/vp/apphome; do
  if [[ ! -d "$directory" ]]; then
    echo "Persistent directory does not exist: $directory" >&2
    exit 1
  fi
done

rollback() {
  exit_code=$?
  trap - ERR
  echo "Deployment failed; restoring the previous container." >&2
  if [[ "$new_container_started" -eq 1 ]]; then
    docker logs --tail 100 "$CONTAINER_NAME" 2>&1 || true
    docker rm --force "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  if [[ "$previous_available" -eq 1 ]] && docker container inspect "$ROLLBACK_CONTAINER" >/dev/null 2>&1; then
    docker rename "$ROLLBACK_CONTAINER" "$CONTAINER_NAME"
    docker start "$CONTAINER_NAME" >/dev/null
  elif [[ "$current_stopped" -eq 1 ]] && docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    docker start "$CONTAINER_NAME" >/dev/null
  fi
  docker logout "$LOGIN_SERVER" >/dev/null 2>&1 || true
  exit "$exit_code"
}
trap rollback ERR

az login --identity --allow-no-subscriptions --output none
token="$(az acr login --name "$ACR_NAME" --expose-token --query accessToken --output tsv)"
if [[ -z "$token" ]]; then
  echo "Managed identity did not receive an ACR token." >&2
  exit 1
fi
printf '%s' "$token" |
  docker login "$LOGIN_SERVER" \
    --username 00000000-0000-0000-0000-000000000000 \
    --password-stdin >/dev/null
unset token

docker pull "$IMAGE"

if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 &&
   docker container inspect "$ROLLBACK_CONTAINER" >/dev/null 2>&1; then
  docker rename "$ROLLBACK_CONTAINER" "$CONTAINER_NAME"
  docker start "$CONTAINER_NAME" >/dev/null
fi

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  docker rm --force "$ROLLBACK_CONTAINER" >/dev/null 2>&1 || true
  docker stop --time 30 "$CONTAINER_NAME" >/dev/null
  current_stopped=1
  docker rename "$CONTAINER_NAME" "$ROLLBACK_CONTAINER"
  previous_available=1
  current_stopped=0
fi

docker run --detach \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --restart unless-stopped \
  --publish 8000:8000 \
  --env-file "$ENV_FILE" \
  --volume /home/azureuser/vp/models:/models \
  --volume /home/azureuser/vp/data:/data \
  --volume /home/azureuser/vp/gallery:/gallery \
  --volume /home/azureuser/vp/results:/results \
  --volume /home/azureuser/vp/apphome:/home/appuser \
  "$IMAGE" >/dev/null
new_container_started=1

healthy=0
for _ in $(seq 1 72); do
  state="$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME")"
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$CONTAINER_NAME")"
  echo "Container state=$state health=$health"

  if [[ "$state" == "running" && "$health" == "healthy" ]] &&
     curl --fail --silent --show-error http://127.0.0.1:8000/docs >/dev/null; then
    healthy=1
    break
  fi
  if [[ "$state" == "exited" || "$state" == "dead" || "$health" == "unhealthy" ]]; then
    break
  fi
  sleep 5
done

if [[ "$healthy" -ne 1 ]]; then
  echo "New container did not become healthy." >&2
  false
fi

trap - ERR
if [[ "$previous_available" -eq 1 ]]; then
  docker rm --force "$ROLLBACK_CONTAINER" >/dev/null 2>&1 || true
fi
docker logout "$LOGIN_SERVER" >/dev/null 2>&1 || true
docker image prune --force >/dev/null 2>&1 || true
echo "Deployment succeeded: $IMAGE"
echo "DEPLOYMENT_RESULT=success"
