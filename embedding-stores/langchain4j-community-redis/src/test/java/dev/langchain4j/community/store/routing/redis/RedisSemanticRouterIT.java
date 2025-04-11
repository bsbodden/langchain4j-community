package dev.langchain4j.community.store.routing.redis;

import static org.assertj.core.api.Assertions.assertThat;
import static org.awaitility.Awaitility.await;

import com.redis.testcontainers.RedisStackContainer;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import java.time.Duration;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;
import redis.clients.jedis.JedisPooled;

@Testcontainers
class RedisSemanticRouterIT {

    private static final DockerImageName REDIS_STACK_IMAGE = DockerImageName.parse("redis/redis-stack:latest");

    @Container
    private static final RedisStackContainer REDIS = new RedisStackContainer(REDIS_STACK_IMAGE);

    private static JedisPooled jedis;
    private static EmbeddingModel embeddingModel;
    private RedisSemanticRouter router;

    @BeforeAll
    static void beforeAll() {
        jedis = new JedisPooled(REDIS.getHost(), REDIS.getFirstMappedPort());
        embeddingModel = new AllMiniLmL6V2EmbeddingModel();
    }

    @AfterAll
    static void afterAll() {
        if (jedis != null) {
            jedis.close();
        }
    }

    @BeforeEach
    void setUp() {
        router = RedisSemanticRouter.builder()
                .redis(jedis)
                .embeddingModel(embeddingModel)
                .prefix("test-router")
                .build();
        
        // Clear any existing routes
        router.clear();
        
        // Add test routes
        Map<String, Object> metadata1 = new HashMap<>();
        metadata1.put("category", "customer");
        metadata1.put("priority", "high");
        
        Route customerRoute = Route.builder()
                .name("customer_support")
                .addReference("I need help with my account")
                .addReference("How do I reset my password?")
                .addReference("I can't login to my account")
                .distanceThreshold(0.15)
                .metadata(metadata1)
                .build();
        
        Map<String, Object> metadata2 = new HashMap<>();
        metadata2.put("category", "technical");
        metadata2.put("priority", "medium");
        
        Route technicalRoute = Route.builder()
                .name("technical_support")
                .addReference("My application is not working")
                .addReference("I'm getting an error message")
                .addReference("The system is slow")
                .distanceThreshold(0.15)
                .metadata(metadata2)
                .build();
        
        Map<String, Object> metadata3 = new HashMap<>();
        metadata3.put("category", "sales");
        metadata3.put("priority", "low");
        
        Route salesRoute = Route.builder()
                .name("sales")
                .addReference("I want to upgrade my subscription")
                .addReference("What are your pricing options?")
                .addReference("Do you offer discounts for annual plans?")
                .distanceThreshold(0.15)
                .metadata(metadata3)
                .build();
        
        router.addRoute(customerRoute);
        router.addRoute(technicalRoute);
        router.addRoute(salesRoute);
        
        // Give Redis time to index
        await().atMost(Duration.ofSeconds(2)).until(() -> router.listRoutes().size() == 3);
    }

    @Test
    void shouldAddAndListRoutes() {
        List<String> routes = router.listRoutes();
        
        assertThat(routes).hasSize(3);
        assertThat(routes).contains("customer_support", "technical_support", "sales");
    }

    @Test
    void shouldRetrieveRouteByName() {
        Route route = router.getRoute("customer_support");
        
        assertThat(route).isNotNull();
        assertThat(route.getName()).isEqualTo("customer_support");
        assertThat(route.getReferences()).hasSize(3);
        assertThat(route.getDistanceThreshold()).isEqualTo(0.15);
        assertThat(route.getMetadata()).containsEntry("category", "customer");
        assertThat(route.getMetadata()).containsEntry("priority", "high");
    }

    @Test
    void shouldRouteToCorrectDestination() {
        // Test customer support query
        List<RouteMatch> matches = router.route("I forgot my password and need to reset it");
        
        assertThat(matches).isNotEmpty();
        RouteMatch bestMatch = matches.get(0);
        assertThat(bestMatch.getRouteName()).isEqualTo("customer_support");
        
        // Test technical support query
        matches = router.route("My application is crashing with an error");
        
        assertThat(matches).isNotEmpty();
        bestMatch = matches.get(0);
        assertThat(bestMatch.getRouteName()).isEqualTo("technical_support");
        
        // Test sales query
        matches = router.route("What is the cost of your premium plan?");
        
        assertThat(matches).isNotEmpty();
        bestMatch = matches.get(0);
        assertThat(bestMatch.getRouteName()).isEqualTo("sales");
    }

    @Test
    void shouldRemoveRoute() {
        boolean removed = router.removeRoute("sales");
        
        assertThat(removed).isTrue();
        assertThat(router.listRoutes()).hasSize(2);
        assertThat(router.listRoutes()).doesNotContain("sales");
        
        // Verify that route no longer exists
        Route route = router.getRoute("sales");
        assertThat(route).isNull();
    }

    @Test
    void shouldClearAllRoutes() {
        router.clear();
        
        assertThat(router.listRoutes()).isEmpty();
    }

    @Test
    void shouldReturnEmptyListForUnknownQuery() {
        // This query should be too dissimilar to match any route
        List<RouteMatch> matches = router.route("XYZ123 completely unrelated query");
        
        assertThat(matches).isEmpty();
    }

    @Test
    void shouldRespectDistanceThreshold() {
        // Add a route with a high threshold (easier to match)
        Route easyRoute = Route.builder()
                .name("easy_match")
                .addReference("Example reference")
                .distanceThreshold(0.9) // Very high threshold
                .build();
        
        // Add a route with a low threshold (harder to match)
        Route hardRoute = Route.builder()
                .name("hard_match")
                .addReference("Example reference")
                .distanceThreshold(0.1) // Very low threshold
                .build();
        
        router.addRoute(easyRoute);
        router.addRoute(hardRoute);
        
        // Wait for indexing
        await().atMost(Duration.ofSeconds(2)).until(() -> router.listRoutes().size() == 5);
        
        // Query that's somewhat related but not exactly matching
        List<RouteMatch> matches = router.route("This is an example");
        
        // Should match the easy route but not the hard route
        boolean hasEasyMatch = matches.stream()
                .anyMatch(match -> match.getRouteName().equals("easy_match"));
        boolean hasHardMatch = matches.stream()
                .anyMatch(match -> match.getRouteName().equals("hard_match"));
        
        assertThat(hasEasyMatch).isTrue();
        assertThat(hasHardMatch).isFalse();
    }

    @Test
    void shouldReturnMultipleMatches() {
        // Add two very similar routes
        Route route1 = Route.builder()
                .name("greetings_1")
                .addReference("Hello, how are you?")
                .distanceThreshold(0.2)
                .build();
        
        Route route2 = Route.builder()
                .name("greetings_2")
                .addReference("Hi, how are you doing?")
                .distanceThreshold(0.2)
                .build();
        
        router.addRoute(route1);
        router.addRoute(route2);
        
        // Wait for indexing
        await().atMost(Duration.ofSeconds(2)).until(() -> router.listRoutes().size() == 5);
        
        // Should match both greeting routes
        List<RouteMatch> matches = router.route("Hey, how are you today?");
        
        assertThat(matches.size()).isGreaterThanOrEqualTo(2);
        
        List<String> matchedRoutes = matches.stream()
                .map(RouteMatch::getRouteName)
                .toList();
        
        assertThat(matchedRoutes).contains("greetings_1", "greetings_2");
    }
}