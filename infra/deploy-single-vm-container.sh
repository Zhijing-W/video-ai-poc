#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="__DEPLOY_IMAGE__"
CONTAINER_NAME="videopoc"
ROLLBACK_CONTAINER="${CONTAINER_NAME}-previous"
ENV_FILE="/home/azureuser/videopoc/.env"
LOGIN_SERVER="${IMAGE%%/*}"
ACR_NAME="${LOGIN_SERVER%%.*}"
IMAGE_REPOSITORY="${IMAGE%:*}"
APP_IMAGE_RETENTION="${APP_IMAGE_RETENTION:-2}"
BUILD_CACHE_RETENTION="${BUILD_CACHE_RETENTION:-168h}"
LEGACY_IMAGE_REF="${LEGACY_IMAGE_REF:-videopoc:gpu}"
ROLLBACK_IMAGE_REF="${ROLLBACK_IMAGE_REF:-videopoc-rollback:previous}"
previous_available=0
previous_image_id=""
current_stopped=0
new_container_started=0

if [[ "$IMAGE" == *DEPLOY_IMAGE* || "$IMAGE" != */*:* ]]; then
  echo "A fully qualified deployment image is required." >&2
  exit 1
fi
if ! [[ "$APP_IMAGE_RETENTION" =~ ^[0-9]+$ ]] ||
   ((APP_IMAGE_RETENTION < 2)); then
  echo "APP_IMAGE_RETENTION must be an integer of at least 2." >&2
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

prune_old_app_images() {
  local current_image_id
  local image_id
  local image_ref
  local legacy_image_id
  local rollback_image_id
  local retained_previous=0
  local previous_limit=$((APP_IMAGE_RETENTION - 1))
  declare -A seen_image_ids=()

  current_image_id="$(
    docker inspect --format '{{.Image}}' "$CONTAINER_NAME"
  )"
  rollback_image_id="$(
    docker image inspect "$ROLLBACK_IMAGE_REF" \
      --format '{{.Id}}' 2>/dev/null || true
  )"
  if [[ -n "$rollback_image_id" &&
        "$rollback_image_id" != "$current_image_id" ]]; then
    retained_previous=1
  fi

  while IFS= read -r image_id; do
    [[ -n "$image_id" ]] || continue
    if [[ -n "${seen_image_ids[$image_id]:-}" ]]; then
      continue
    fi
    seen_image_ids["$image_id"]=1

    if [[ "$image_id" == "$current_image_id" ||
          "$image_id" == "$rollback_image_id" ]]; then
      continue
    fi
    if ((retained_previous < previous_limit)); then
      retained_previous=$((retained_previous + 1))
      continue
    fi

    while IFS= read -r image_ref; do
      if [[ "$image_ref" == "${IMAGE_REPOSITORY}:"* ]]; then
        docker image rm "$image_ref" >/dev/null 2>&1 ||
          echo "Warning: unable to remove old image tag $image_ref" >&2
      fi
    done < <(
      docker image inspect "$image_id" \
        --format '{{range .RepoTags}}{{println .}}{{end}}'
    )
  done < <(
    docker image ls "$IMAGE_REPOSITORY" \
      --no-trunc \
      --format '{{.ID}}'
  )

  legacy_image_id="$(
    docker image inspect "$LEGACY_IMAGE_REF" \
      --format '{{.Id}}' 2>/dev/null || true
  )"
  if [[ -n "$legacy_image_id" &&
        "$legacy_image_id" != "$current_image_id" &&
        "$legacy_image_id" != "$previous_image_id" ]]; then
    docker image rm "$LEGACY_IMAGE_REF" >/dev/null 2>&1 ||
      echo "Warning: unable to remove legacy image $LEGACY_IMAGE_REF" >&2
  fi

  docker image prune --force >/dev/null 2>&1 ||
    echo "Warning: unable to prune dangling images." >&2
  docker builder prune \
    --force \
    --filter "until=${BUILD_CACHE_RETENTION}" >/dev/null 2>&1 ||
    echo "Warning: unable to prune expired build cache." >&2
}

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
  previous_image_id="$(
    docker inspect --format '{{.Image}}' "$CONTAINER_NAME"
  )"
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
  --env MODEL_ROOT=/models \
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
     curl --fail --silent --show-error http://127.0.0.1:8000/health >/dev/null; then
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

current_image_id="$(
  docker inspect --format '{{.Image}}' "$CONTAINER_NAME"
)"
if [[ "$previous_available" -eq 1 &&
      -n "$previous_image_id" &&
      "$previous_image_id" != "$current_image_id" ]]; then
  docker image tag "$previous_image_id" "$ROLLBACK_IMAGE_REF"
fi

trap - ERR
if [[ "$previous_available" -eq 1 ]]; then
  docker rm --force "$ROLLBACK_CONTAINER" >/dev/null 2>&1 || true
fi
docker logout "$LOGIN_SERVER" >/dev/null 2>&1 || true
prune_old_app_images
echo "Deployment succeeded: $IMAGE"
echo "DEPLOYMENT_RESULT=success"
