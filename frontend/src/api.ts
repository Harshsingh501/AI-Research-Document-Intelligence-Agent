// One small typed wrapper around fetch. Every call to the backend goes through
// `request`, so error handling and the base URL live in exactly one place.

import type {
  ConversationDetail,
  ConversationOut,
  DocumentOut,
  SendMessageRequest,
  SendMessageResponse,
  SourceOut,
} from "./types";

export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    // FastAPI puts human-readable errors in {"detail": ...}; fall back to status text
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON body */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

// --- Documents ---------------------------------------------------------------

export const listDocuments = () =>
  request<{ items: DocumentOut[]; total: number }>("/documents").then((r) => r.items);

export function uploadDocument(file: File) {
  // multipart/form-data: the browser sets the Content-Type boundary itself,
  // so we must NOT set the header manually.
  const form = new FormData();
  form.append("file", file);
  return request<DocumentOut>("/documents", { method: "POST", body: form });
}

export const deleteDocument = (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" });

// --- Conversations -----------------------------------------------------------

export const listConversations = () =>
  request<{ items: ConversationOut[]; total: number }>("/conversations").then((r) => r.items);

export const createConversation = () => request<ConversationOut>("/conversations", json({}));

export const getConversation = (id: string) => request<ConversationDetail>(`/conversations/${id}`);

export const deleteConversation = (id: string) =>
  request<void>(`/conversations/${id}`, { method: "DELETE" });

export const sendMessage = (id: string, body: SendMessageRequest) =>
  request<SendMessageResponse>(`/conversations/${id}/messages`, json(body));


// --- Streaming ---------------------------------------------------------------------

export interface StreamStatus {
  stage: "condensing" | "retrieving" | "generating" | "agent";
  standalone_question?: string;
}

export interface StreamHandlers {
  onStatus?: (s: StreamStatus) => void;
  onSources?: (s: SourceOut[]) => void;
  onToken?: (text: string) => void;
}

// EventSource only does GET, so we read the SSE body from a POST by hand.
// Blocks are separated by a blank line; each has `event:` and `data:` lines.
export async function sendMessageStream(
  id: string,
  body: SendMessageRequest,
  h: StreamHandlers,
): Promise<SendMessageResponse> {
  const res = await fetch(`${API_URL}/conversations/${id}/messages/stream`, json(body));
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: SendMessageResponse | null = null;

  const handle = (block: string) => {
    let event = "";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (!event) return;
    const payload = data ? JSON.parse(data) : null;
    switch (event) {
      case "status":
        h.onStatus?.(payload as StreamStatus);
        break;
      case "sources":
        h.onSources?.(payload as SourceOut[]);
        break;
      case "token":
        h.onToken?.((payload as { text: string }).text);
        break;
      case "done":
        result = payload as SendMessageResponse;
        break;
      case "error":
        throw new ApiError(500, (payload as { detail: string }).detail);
    }
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      handle(buffer.slice(0, idx));
      buffer = buffer.slice(idx + 2);
    }
  }
  if (!result) throw new ApiError(500, "The stream ended without a result.");
  return result;
}
