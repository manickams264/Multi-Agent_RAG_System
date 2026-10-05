# LangGraph Checkpointing Notes

LangGraph supports checkpointing so graph state can be inspected after each node.
MemorySaver stores checkpoints in process memory. For durable runs, a SQLite
checkpointer can persist state to disk. State transition logs should record the
node name, UTC timestamp, and a short summary of fields that changed.

## Parent Document Retrieval (optional technique)

Parent-document retrieval embeds small child chunks for precise search, but
returns larger parent passages to the generator so answers have fuller context.

## Step-Back Prompting (optional technique)

Step-back prompting first retrieves broader background context, then retrieves
specific details, and combines both before generation.
