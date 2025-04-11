# LangChain4j Community: Redis Integration

This module provides Redis integration for LangChain4j, offering the following features:

- Redis as a vector store for embeddings with filtering support
- Redis-based chat memory store for conversation history
- Redis-based caching for LLM responses (exact matching)
- Redis-based semantic caching using vector similarity
- Redis-based semantic routing for directing queries to appropriate handlers

## Redis as an Embedding Store

The `RedisEmbeddingStore` class provides Redis-based vector storage and retrieval for embeddings. It uses Redis Stack with RediSearch for efficient vector similarity search.

### Features

- Support for exact match and vector similarity search
- Support for metadata filtering using Redis JSON
- Both FLAT and HNSW indexing algorithms
- Configurable distance metrics (COSINE, IP, L2)

## Redis as a Chat Memory Store

The `RedisChatMemoryStore` class provides Redis-based storage for conversation history. It uses Redis JSON to store and retrieve messages.

## Redis for LLM Caching

### Exact Matching Cache

The `RedisCache` class provides Redis-based caching for LLM responses using exact matching.

### Semantic Caching

The `RedisSemanticCache` class provides semantic caching for LLM responses using vector similarity. This allows finding semantically similar prompts and reusing their responses.

## Redis for Semantic Routing

The `RedisSemanticRouter` class provides semantic routing capabilities, allowing you to direct queries to appropriate handlers based on semantic similarity.

### Features

- Define routes with reference texts and distance thresholds
- Route text inputs to the most semantically similar routes
- Retrieve route metadata for additional context

## Spring Boot Integration

This module provides Spring Boot auto-configuration for all Redis features:

- `RedisEmbeddingStoreAutoConfiguration`: Auto-configuration for `RedisEmbeddingStore`
- `RedisCacheAutoConfiguration`: Auto-configuration for `RedisCache`
- `RedisSemanticCacheAutoConfiguration`: Auto-configuration for `RedisSemanticCache`
- `RedisSemanticRouterAutoConfiguration`: Auto-configuration for `RedisSemanticRouter`

### Configuration Properties

#### Redis Embedding Store Properties

```properties
langchain4j.community.redis.embedding-store.host=localhost
langchain4j.community.redis.embedding-store.port=6379
langchain4j.community.redis.embedding-store.user=default
langchain4j.community.redis.embedding-store.password=password
langchain4j.community.redis.embedding-store.index-type=FLAT
langchain4j.community.redis.embedding-store.metric-type=COSINE
langchain4j.community.redis.embedding-store.vector-dim=1536
langchain4j.community.redis.embedding-store.index-name=my-index
langchain4j.community.redis.embedding-store.prefix=vector
```

#### Redis Cache Properties

```properties
langchain4j.community.redis.cache.enabled=true
langchain4j.community.redis.cache.ttl=3600
langchain4j.community.redis.cache.prefix=exact-cache
```

#### Redis Semantic Cache Properties

```properties
langchain4j.community.redis.semantic-cache.enabled=true
langchain4j.community.redis.semantic-cache.ttl=3600
langchain4j.community.redis.semantic-cache.prefix=semantic-cache
langchain4j.community.redis.semantic-cache.similarity-threshold=0.2
```

#### Redis Semantic Router Properties

```properties
langchain4j.community.redis.semantic-router.enabled=true
langchain4j.community.redis.semantic-router.prefix=semantic-router
langchain4j.community.redis.semantic-router.max-results=5
```

## Examples

### Using RedisEmbeddingStore

```java
// Create Redis client
JedisPooled jedis = new JedisPooled("localhost", 6379);

// Create embedding store
RedisEmbeddingStore embeddingStore = RedisEmbeddingStore.builder()
    .redisClient(jedis)
    .indexName("my-index")
    .vectorDimension(1536)
    .build();

// Add embeddings
embeddingStore.add(Embedding.from(vector1), metadata1, id1);
embeddingStore.add(Embedding.from(vector2), metadata2, id2);

// Find similar embeddings
List<EmbeddingMatch<Metadata>> matches = embeddingStore.findRelevant(
    Embedding.from(queryVector), 5);
```

### Using RedisCache (Exact Matching)

```java
// Create Redis client
JedisPooled jedis = new JedisPooled("localhost", 6379);

// Create cache
RedisCache cache = RedisCache.builder()
    .redis(jedis)
    .ttl(3600) // 1 hour TTL
    .prefix("my-cache")
    .build();

// Look up cached response
Response<?> cachedResponse = cache.lookup(prompt, modelString);

if (cachedResponse == null) {
    // Get response from model and cache it
    Response<?> newResponse = model.generate(prompt);
    cache.update(prompt, modelString, newResponse);
    return newResponse;
} else {
    return cachedResponse;
}
```

### Using RedisSemanticCache

```java
// Create Redis client
JedisPooled jedis = new JedisPooled("localhost", 6379);

// Create embedding model
EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

// Create semantic cache
RedisSemanticCache semanticCache = RedisSemanticCache.builder()
    .redis(jedis)
    .embeddingModel(embeddingModel)
    .ttl(3600) // 1 hour TTL
    .prefix("semantic-cache")
    .similarityThreshold(0.2f)
    .build();

// Look up cached response by semantic similarity
Response<?> cachedResponse = semanticCache.lookup(prompt, modelString);

if (cachedResponse == null) {
    // Get response from model and cache it
    Response<?> newResponse = model.generate(prompt);
    semanticCache.update(prompt, modelString, newResponse);
    return newResponse;
} else {
    return cachedResponse;
}
```

### Using RedisSemanticRouter

```java
// Create Redis client
JedisPooled jedis = new JedisPooled("localhost", 6379);

// Create embedding model
EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

// Create semantic router
RedisSemanticRouter router = RedisSemanticRouter.builder()
    .redis(jedis)
    .embeddingModel(embeddingModel)
    .prefix("semantic-router")
    .maxResults(5)
    .build();

// Define routes
Route customerRoute = Route.builder()
    .name("customer_support")
    .addReference("I need help with my account")
    .addReference("How do I reset my password?")
    .distanceThreshold(0.2)
    .addMetadata("department", "customer-service")
    .build();

Route technicalRoute = Route.builder()
    .name("technical_support")
    .addReference("I'm getting an error message")
    .addReference("The system is not working")
    .distanceThreshold(0.2)
    .addMetadata("department", "engineering")
    .build();

// Add routes to router
router.addRoute(customerRoute);
router.addRoute(technicalRoute);

// Route a query
List<RouteMatch> matches = router.route("I can't log into my account");

// Process the best match
if (!matches.isEmpty()) {
    RouteMatch bestMatch = matches.get(0);
    System.out.println("Routed to: " + bestMatch.getRouteName());
    System.out.println("Similarity: " + bestMatch.getDistance());
    System.out.println("Metadata: " + bestMatch.getMetadata());
}
```

## Spring Boot Example

```java
@SpringBootApplication
public class MyApplication {

    public static void main(String[] args) {
        SpringApplication.run(MyApplication.class, args);
    }

    @Bean
    public EmbeddingModel embeddingModel() {
        return new AllMiniLmL6V2EmbeddingModel();
    }

    @Bean
    public LanguageModel languageModel() {
        return OpenAiLanguageModel.builder()
            .apiKey(System.getenv("OPENAI_API_KEY"))
            .modelName("gpt-3.5-turbo")
            .build();
    }
}
```

```properties
# Redis connection
langchain4j.community.redis.embedding-store.host=localhost
langchain4j.community.redis.embedding-store.port=6379

# Enable Redis features
langchain4j.community.redis.cache.enabled=true
langchain4j.community.redis.semantic-cache.enabled=true
langchain4j.community.redis.semantic-router.enabled=true
```

## Requirements

- Redis Stack >= 6.2.0 with RediSearch and RedisJSON modules
- Java 17 or higher