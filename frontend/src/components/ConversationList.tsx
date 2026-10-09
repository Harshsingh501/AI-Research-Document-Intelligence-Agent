import type { ConversationOut } from "../types";

interface Props {
  conversations: ConversationOut[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

export function ConversationList({ conversations, selectedId, onSelect, onNew, onDelete }: Props) {
  return (
    <section className="panel conversations">
      <header className="panel-header">
        <h2>Chats</h2>
        <button className="btn small" onClick={onNew}>
          + New
        </button>
      </header>
      {conversations.length === 0 && <p className="muted">No chats yet.</p>}
      <ul className="conv-list">
        {conversations.map((c) => (
          <li key={c.id} className={c.id === selectedId ? "selected" : ""}>
            <button className="conv-title" onClick={() => onSelect(c.id)}>
              {c.title ?? "New chat"}
            </button>
            <button
              className="icon"
              title="Delete chat"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(c.id);
              }}
            >
              ×
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
