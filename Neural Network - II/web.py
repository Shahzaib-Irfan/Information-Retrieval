import streamlit as st
import math
import random

class NeuralNetwork:
    def __init__(self, input_size=4, hidden_size=5, output_size=1):
        random.seed(42)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = [
            [random.uniform(-1, 1) for _ in range(hidden_size)]
            for _ in range(input_size)
        ]
        
        self.weights_hidden_output = [
            [random.uniform(-1, 1) for _ in range(output_size)]
            for _ in range(hidden_size)
        ]
        
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(output_size)]
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def forward_propagation(self, input_vector):
        hidden_layer = []
        for j in range(self.hidden_size):
            neuron_sum = sum(
                input_vector[i] * self.weights_input_hidden[i][j] 
                for i in range(self.input_size)
            ) + self.bias_hidden[j]
            
            hidden_layer.append(self.sigmoid(neuron_sum))
        
        output_layer = []
        for j in range(self.output_size):
            neuron_sum = sum(
                hidden_layer[i] * self.weights_hidden_output[i][j] 
                for i in range(self.hidden_size)
            ) + self.bias_output[j]
            
            output_layer.append(self.sigmoid(neuron_sum))
        
        return {
            'hidden_layer': hidden_layer,
            'output_layer': output_layer
        }
    
    def train(self, training_data, labels, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            total_error = 0
            
            for input_vector, target in zip(training_data, labels):
                forward_result = self.forward_propagation(input_vector)
                
                output_errors = [
                    (target[j] - forward_result['output_layer'][j]) 
                    for j in range(self.output_size)
                ]
                
                total_error += sum(error**2 for error in output_errors)
        
        return total_error
    
    def predict(self, input_vector):
        return self.forward_propagation(input_vector)['output_layer']

class TechTrendIRProcessor:
    def __init__(self):
        self.semantic_dictionary = {
            "ai": ["machine learning", "artificial intelligence", "neural networks", "deep learning"],
            "blockchain": ["cryptocurrency", "decentralized", "web3", "smart contracts"],
            "cloud": ["cloud computing", "distributed systems", "serverless", "infrastructure"],
            "cybersecurity": ["network security", "data protection", "encryption", "privacy"],
            "startups": ["innovation", "tech companies", "entrepreneurship", "venture capital"]
        }
        
        self.articles = [
            {
                "title": "The Rise of Generative AI in Enterprise Solutions",
                "content": "Generative AI is transforming enterprise software, offering unprecedented capabilities in automation, content creation, and decision support systems.",
                "keywords": ["ai", "machine learning", "enterprise"],
                "features": [0.7, 0.5, 0.6, 0.4]
            },
            {
                "title": "Blockchain Beyond Cryptocurrency: Real-World Applications",
                "content": "Blockchain technology is expanding beyond cryptocurrency, finding applications in supply chain, healthcare, and digital identity verification.",
                "keywords": ["blockchain", "decentralized", "technology"],
                "features": [0.6, 0.4, 0.5, 0.3]
            },
            {
                "title": "Cloud Computing: The Backbone of Modern Digital Infrastructure",
                "content": "Cloud computing continues to revolutionize how businesses manage and scale their technological infrastructure, offering flexibility and cost-efficiency.",
                "keywords": ["cloud", "infrastructure", "technology"],
                "features": [0.8, 0.6, 0.7, 0.5]
            },
            {
                "title": "Emerging Cybersecurity Threats in the AI Era",
                "content": "As AI advances, new cybersecurity challenges emerge, requiring innovative approaches to network protection and data privacy.",
                "keywords": ["cybersecurity", "ai", "network security"],
                "features": [0.9, 0.7, 0.8, 0.6]
            }
        ]
        
        self.nn = NeuralNetwork(input_size=4, hidden_size=5, output_size=1)
        self._prepare_training_data()
    
    def _prepare_training_data(self):
        X_train = [article['features'] for article in self.articles]
        
        def calculate_relevance(article):
            return [min(1.0, len(set(article['keywords'])) / 3.0)]
        
        y_train = [calculate_relevance(article) for article in self.articles]
        
        self.nn.train(X_train, y_train, epochs=1000)
    
    def preprocess_text(self, query):
        return [word.lower() for word in query.split()]
    
    def semantic_understanding(self, keywords):
        semantic_map = {}
        for word in keywords:
            related_terms = self.semantic_dictionary.get(word, [])
            semantic_map[word] = related_terms
        return semantic_map
    
    def query_expansion(self, keywords, semantic_map):
        expanded_terms = set(keywords)
        for word in keywords:
            expanded_terms.update(semantic_map.get(word, []))
        return list(expanded_terms)
    
    def extract_features(self, expanded_terms):
        return [
            len(expanded_terms) / 10.0,
            len(set(expanded_terms)) / len(expanded_terms),
            sum(len(term) for term in expanded_terms) / len(expanded_terms),
            len([term for term in expanded_terms if any(t in term for t in self.semantic_dictionary)])
        ]
    
    def search_and_retrieve(self, expanded_terms):
        features = self.extract_features(expanded_terms)
        
        matching_articles = []
        for article in self.articles:
            if any(term.lower() in ' '.join(article['keywords']).lower() or 
                   term.lower() in article.get('content', '').lower() 
                   for term in expanded_terms):
                relevance_score = self.nn.predict(features)[0]
                
                matching_articles.append({
                    'article': article,
                    'relevance_score': relevance_score
                })
        
        return sorted(matching_articles, key=lambda x: x['relevance_score'], reverse=True)
    
    def process_query(self, query):
        keywords = self.preprocess_text(query)
        semantic_map = self.semantic_understanding(keywords)
        expanded_terms = self.query_expansion(keywords, semantic_map)
        matching_articles = self.search_and_retrieve(expanded_terms)
        
        return matching_articles

def main():
    st.set_page_config(page_title="Tech Trend Insights", page_icon="🚀", layout="wide")
    
    # Title and Introduction
    st.title("🌐 Tech Trend Information Retrieval")
    st.markdown("""
    ### Discover Cutting-Edge Tech Insights
    Explore technology trends using our advanced semantic search and neural network-powered relevance ranking.
    """)
    
    # Sidebar for additional context
    st.sidebar.title("🤖 About the Tool")
    st.sidebar.info("""
    This app uses:
    - Semantic query expansion
    - Neural network relevance scoring
    - Advanced information retrieval techniques
    """)
    
    # Initialize the IR Processor
    if 'ir_processor' not in st.session_state:
        st.session_state.ir_processor = TechTrendIRProcessor()
    
    # Query Input
    query = st.text_input("Enter your technology query", placeholder="e.g., AI enterprise solutions")
    
    # Search Button
    if st.button("Search Tech Trends") or query:
        if query:
            # Process the query
            results = st.session_state.ir_processor.process_query(query)
            
            # Display Results
            if results:
                st.subheader("🔍 Matching Articles")
                for result in results:
                    article = result['article']
                    relevance = result['relevance_score']
                    
                    # Create an expandable card for each article
                    with st.expander(f"{article['title']} (Relevance: {relevance:.2f})", expanded=False):
                        st.markdown(f"**Keywords:** {', '.join(article['keywords'])}")
                        st.write(article['content'])
                        st.progress(relevance, f"Relevance Score: {relevance:.2f}")
            else:
                st.warning("No matching articles found.")
    
    # Semantic Dictionary Visualization
    st.sidebar.subheader("🔗 Semantic Relationships")
    selected_term = st.sidebar.selectbox(
        "Explore Related Terms", 
        list(st.session_state.ir_processor.semantic_dictionary.keys())
    )
    st.sidebar.write("Related Terms:")
    st.sidebar.write(", ".join(st.session_state.ir_processor.semantic_dictionary[selected_term]))

if __name__ == "__main__":
    main()