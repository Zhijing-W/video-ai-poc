{{- define "video-poc.name" -}}
video-poc
{{- end }}

{{- define "video-poc.labels" -}}
app.kubernetes.io/name: {{ include "video-poc.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end }}

{{- define "video-poc.selectorLabels" -}}
app.kubernetes.io/name: {{ include "video-poc.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "video-poc.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "video-poc.name" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
