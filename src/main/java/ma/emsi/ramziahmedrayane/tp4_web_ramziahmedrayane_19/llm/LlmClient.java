package ma.emsi.ramziahmedrayane.tp4_web_ramziahmedrayane_19.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.enterprise.context.Dependent;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Dependent
public class LlmClient implements Serializable {

    private String systemRole;
    private Assistant assistant;
    private ChatMemory chatMemory;

    public LlmClient() {
        String cle = System.getenv("GEMINI_KEY");

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(20);

        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        EmbeddingStore<TextSegment> storeIA = chargerDocument(
                "Fine Tuning & RAG.pdf", embeddingModel);

        EmbeddingStore<TextSegment> storeArchitecture = chargerDocument(
                "35_2 Stockstill_pp 65-82.pdf", embeddingModel);

        ContentRetriever retrieverIA =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(storeIA)
                        .embeddingModel(embeddingModel)
                        .maxResults(4)
                        .minScore(0.5)
                        .build();

        ContentRetriever retrieverArchitecture =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(storeArchitecture)
                        .embeddingModel(embeddingModel)
                        .maxResults(4)
                        .minScore(0.5)
                        .build();

        Map<ContentRetriever, String> sources = new LinkedHashMap<>();

        sources.put(retrieverIA,
                "Support de cours sur l'intelligence artificielle, "
                        + "les modèles de langage, le fine-tuning et le RAG.");

        sources.put(retrieverArchitecture,
                "Article sur l'histoire et l'architecture de la Koutoubia "
                        + "à Marrakech, les deux mosquées et la période almohade.");

        QueryRouter queryRouter =
                new LanguageModelQueryRouter(model, sources);

        QueryTransformer queryTransformer =
                CompressingQueryTransformer.builder()
                        .chatModel(model)
                        .build();

        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryTransformer(queryTransformer)
                        .queryRouter(queryRouter)
                        .build();

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(this.chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();
    }

    public void setSystemRole(String systemRole) {
        this.systemRole = systemRole;
        this.chatMemory.clear();
        this.chatMemory.add(SystemMessage.from(this.systemRole));
    }

    public String envoyerRequete(String question) {
        return this.assistant.chat(question);
    }

    private static EmbeddingStore<TextSegment> chargerDocument(
            String nomFichier, EmbeddingModel embeddingModel) {

        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();

        Document document = ClassPathDocumentLoader.loadDocument(
                nomFichier, parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(1500, 200);
        List<TextSegment> segments = splitter.split(document);

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        return store;
    }
}