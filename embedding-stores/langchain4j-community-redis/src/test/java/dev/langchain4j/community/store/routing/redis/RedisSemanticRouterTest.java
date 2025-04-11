package dev.langchain4j.community.store.routing.redis;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import redis.clients.jedis.JedisPooled;
import redis.clients.jedis.json.Path;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class RedisSemanticRouterTest {

    private JedisPooled redis;
    private EmbeddingModel embeddingModel;
    private RedisSemanticRouter router;

    @BeforeEach
    void setUp() {
        redis = mock(JedisPooled.class);
        embeddingModel = mock(EmbeddingModel.class);
        
        // Mock embedding model
        Embedding mockEmbedding = mock(Embedding.class);
        when(mockEmbedding.vector()).thenReturn(new float[]{0.1f, 0.2f, 0.3f});
        Response<Embedding> embeddingResponse = Response.from(mockEmbedding);
        when(embeddingModel.embed(any(String.class))).thenReturn(embeddingResponse);
        
        // Create the router
        router = RedisSemanticRouter.builder()
                .redis(redis)
                .embeddingModel(embeddingModel)
                .build();
    }

    @Test
    void shouldRequireRedisAndEmbeddingModel() {
        assertThatThrownBy(() -> RedisSemanticRouter.builder().embeddingModel(embeddingModel).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Redis client is required");

        assertThatThrownBy(() -> RedisSemanticRouter.builder().redis(redis).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Embedding model is required");
    }

    @Test
    void shouldBuildWithDefaults() {
        RedisSemanticRouter router = RedisSemanticRouter.builder()
                .redis(redis)
                .embeddingModel(embeddingModel)
                .build();

        assertThat(router).isNotNull();
    }

    @Test
    void shouldBuildWithCustomSettings() {
        RedisSemanticRouter router = RedisSemanticRouter.builder()
                .redis(redis)
                .embeddingModel(embeddingModel)
                .prefix("custom-router")
                .maxResults(10)
                .build();

        assertThat(router).isNotNull();
    }

    @Test
    void shouldValidateRouteOnAdd() {
        assertThatThrownBy(() -> router.addRoute(null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Route cannot be null");
    }

    @Test
    void shouldValidateRouteNameOnRemove() {
        assertThatThrownBy(() -> router.removeRoute(null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Route name cannot be null");

        assertThatThrownBy(() -> router.removeRoute(""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Route name cannot be null or empty");
    }

    @Test
    void shouldValidateTextOnRoute() {
        assertThatThrownBy(() -> router.route(null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Text cannot be null");

        assertThatThrownBy(() -> router.route(""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Text cannot be null or empty");
    }

    @Test
    void shouldValidateRouteNameOnGetRoute() {
        assertThatThrownBy(() -> router.getRoute(null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Route name cannot be null");

        assertThatThrownBy(() -> router.getRoute(""))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Route name cannot be null or empty");
    }

    @Test
    void shouldReturnEmptyListForNoRoutes() {
        List<RouteMatch> matches = router.route("test query");
        assertThat(matches).isEmpty();
    }
}