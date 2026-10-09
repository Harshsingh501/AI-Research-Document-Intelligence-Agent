// Mirrors the Pydantic schemas in backend/app/schemas/. If the API changes,
// change these too — TypeScript will then point at every place that breaks.

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export interface DocumentOut {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  page_count: number | null;
  status: DocumentStatus;
  error_message: string | null;
  created_at: string;
}

export interface SourceOut {
  index: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number | null;
  score: number;
  excerpt: string;
}

export type Confidence = "high" | "medium" | "low" | "none";
export type Mode = "ask" | "agent";
export type MessageRole = "user" | "assistant";

export interface AgentStep {
  tool: string;
  arguments: Record<string, unknown>;
}

export interface MessageMeta {
  mode?: Mode;
  confidence?: Confidence;
  citations?: number[];
  short_circuited?: boolean;
  fallback_used?: boolean;
  fallback_reason?: string | null;
  steps?: AgentStep[];
  standalone_question?: string;
}

export interface MessageOut {
  id: string;
  role: MessageRole;
  content: string;
  sources: SourceOut[] | null;
  meta: MessageMeta | null;
  created_at: string;
}

export interface ConversationOut {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends ConversationOut {
  messages: MessageOut[];
}

export interface SendMessageRequest {
  question: string;
  mode: Mode;
  document_id?: string | null;
  top_k?: number;
}

export interface SendMessageResponse {
  user_message: MessageOut;
  assistant_message: MessageOut;
  standalone_question: string;
}
