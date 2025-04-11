package dev.langchain4j.community.store.cache.redis;

import static org.assertj.core.api.Assertions.assertThat;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import redis.clients.jedis.JedisPooled;

/**
 * Integration test for RedisSemanticCache with TestContainers.
 *
 * This test requires Docker to run Redis in a container.
 */
@Testcontainers
public class RedisSemanticCacheIT {

    private static final int REDIS_PORT = 6379;

    @Container
    private final GenericContainer<?> redis = new GenericContainer<>("redis/redis-stack:latest")
            .withExposedPorts(REDIS_PORT);

    private JedisPooled jedisPooled;
    private RedisSemanticCache semanticCache;
    private EmbeddingModel embeddingModel;

    @BeforeEach
    public void setUp() {
        String host = redis.getHost();
        Integer port = redis.getMappedPort(REDIS_PORT);

        jedisPooled = new JedisPooled(host, port);
        embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        semanticCache = new RedisSemanticCache(jedisPooled, embeddingModel, 3600, "test-cache", 0.6f);
    }

    @AfterEach
    public void tearDown() {
        semanticCache.clear();
        semanticCache.close();
    }

    @Test
    public void should_find_semantically_similar_response() {
        // given
        String prompt1 = "What is the capital of France?";
        String prompt2 = "Tell me about the capital city of France";
        String llmString = "test-llm";
        
        Response<String> response = new Response<>(
                "Paris is the capital of France.", 
                new TokenUsage(10, 20, 30), 
                null);

        // when - store the response for prompt1
        semanticCache.update(prompt1, llmString, response);

        // then - should be able to retrieve it with a semantically similar prompt
        Response<?> result = semanticCache.lookup(prompt2, llmString);
        assertThat(result).isNotNull();
        assertThat(result.content()).isEqualTo("Paris is the capital of France.");
    }

    @Test
    public void should_not_find_semantically_different_response() {
        // given
        String prompt1 = "What is the capital of France?";
        String prompt2 = "What is the population of Germany?"; // Different semantic meaning
        String llmString = "test-llm";
        
        Response<String> response = new Response<>(
                "Paris is the capital of France.", 
                new TokenUsage(10, 20, 30), 
                null);

        // when - store the response for prompt1
        semanticCache.update(prompt1, llmString, response);

        // then - should not find a response for a semantically different prompt
        Response<?> result = semanticCache.lookup(prompt2, llmString);
        assertThat(result).isNull();
    }

    @Test
    public void should_respect_llm_isolation() {
        // given
        String prompt = "What is the capital of France?";
        String llmString1 = "test-llm-1";
        String llmString2 = "test-llm-2";
        
        Response<String> response = new Response<>(
                "Paris is the capital of France.", 
                new TokenUsage(10, 20, 30), 
                null);

        // when - store the response for llmString1
        semanticCache.update(prompt, llmString1, response);

        // then - should not find it when using a different llmString
        Response<?> result = semanticCache.lookup(prompt, llmString2);
        assertThat(result).isNull();
    }
}