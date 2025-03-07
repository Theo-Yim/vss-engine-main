## GraphRAG

General flow:
`text -LLM-> Cypher -Neo4j-> Response`

## Database
Currently Neo4j works better but needs commercial license.
Memgraph is a close alternative with BSD license BUT it has certain features of the OpenCypher protocol not yet implemented.
LLMs are more familiar with generating Neo4j cypher.

## LLMs
- GPT4o excels at Cypher generation.
- LLama-70b works decently close.
- [`text2cypher`](https://huggingface.co/tomasonjo/text2cypher-demo-16bit) is a LLama3-8b finetuned model.

## Browser

Enter the URL, username and password in this website for visualization and interacting with Neo4j.
`https://workspace-preview.neo4j.io/workspace/query`

## Common Cypher queries

- View whole graph
```cypher
MATCH (n)
OPTIONAL MATCH (n)-[r]-(m)
return n, r ,m
```
- Export whole graph
```cypher
CALL apoc.export.cypher.all(null, {format: 'plain', stream: true})
YIELD cypherStatements
RETURN cypherStatements
```
- Merge nodes with given ids
```cypher
MATCH (a), (b)
where id(a) = 4 and id(b) = 41
CALL apoc.refactor.mergeNodes([a, b]) YIELD node
RETURN node
```
- Delete whole graph
```cypher
match (n) DETACH DELETE n
```
