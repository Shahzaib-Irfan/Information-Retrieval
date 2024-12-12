import math
import random

class NeuralNetwork:
    def __init__(self, input_size=4, hidden_size=5, output_size=1):
        """
        Custom neural network implementation from scratch
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of output neurons
        """
        # Set random seed for reproducibility
        random.seed(42)
        
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weight matrices
        self.weights_input_hidden = [
            [random.uniform(-1, 1) for _ in range(hidden_size)]
            for _ in range(input_size)
        ]
        
        self.weights_hidden_output = [
            [random.uniform(-1, 1) for _ in range(output_size)]
            for _ in range(hidden_size)
        ]
        
        # Bias vectors
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(output_size)]
    
    def sigmoid(self, x):
        """
        Sigmoid activation function
        
        Args:
            x (float): Input value
        
        Returns:
            float: Sigmoid activated output
        """
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Derivative of sigmoid activation function
        
        Args:
            x (float): Sigmoid activated value
        
        Returns:
            float: Derivative value
        """
        return x * (1 - x)
    
    def forward_propagation(self, input_vector):
        """
        Forward propagation through the neural network
        
        Args:
            input_vector (list): Input feature vector
        
        Returns:
            dict: Hidden and output layer activations
        """
        # Hidden layer calculations
        hidden_layer = []
        for j in range(self.hidden_size):
            # Compute weighted sum
            neuron_sum = sum(
                input_vector[i] * self.weights_input_hidden[i][j] 
                for i in range(self.input_size)
            ) + self.bias_hidden[j]
            
            # Apply activation
            hidden_layer.append(self.sigmoid(neuron_sum))
        
        # Output layer calculations
        output_layer = []
        for j in range(self.output_size):
            # Compute weighted sum
            neuron_sum = sum(
                hidden_layer[i] * self.weights_hidden_output[i][j] 
                for i in range(self.hidden_size)
            ) + self.bias_output[j]
            
            # Apply activation
            output_layer.append(self.sigmoid(neuron_sum))
        
        return {
            'hidden_layer': hidden_layer,
            'output_layer': output_layer
        }
    
    def train(self, training_data, labels, learning_rate=0.1, epochs=1000):
        """
        Train the neural network using backpropagation
        
        Args:
            training_data (list): List of input feature vectors
            labels (list): Corresponding target labels
            learning_rate (float): Learning rate for weight updates
            epochs (int): Number of training iterations
        """
        for epoch in range(epochs):
            total_error = 0
            
            for input_vector, target in zip(training_data, labels):
                # Forward propagation
                forward_result = self.forward_propagation(input_vector)
                
                # Compute output layer error
                output_errors = [
                    (target[j] - forward_result['output_layer'][j]) * 
                    self.sigmoid_derivative(forward_result['output_layer'][j])
                    for j in range(self.output_size)
                ]
                
                # Compute hidden layer error
                hidden_errors = [
                    sum(output_errors[k] * self.weights_hidden_output[j][k] 
                        for k in range(self.output_size)) * 
                    self.sigmoid_derivative(forward_result['hidden_layer'][j])
                    for j in range(self.hidden_size)
                ]
                
                # Update output layer weights and biases
                for j in range(self.hidden_size):
                    for k in range(self.output_size):
                        self.weights_hidden_output[j][k] += learning_rate * (
                            output_errors[k] * forward_result['hidden_layer'][j]
                        )
                
                for k in range(self.output_size):
                    self.bias_output[k] += learning_rate * output_errors[k]
                
                # Update hidden layer weights and biases
                for i in range(self.input_size):
                    for j in range(self.hidden_size):
                        self.weights_input_hidden[i][j] += learning_rate * (
                            hidden_errors[j] * input_vector[i]
                        )
                
                for j in range(self.hidden_size):
                    self.bias_hidden[j] += learning_rate * hidden_errors[j]
                
                # Compute total error
                total_error += sum((target[j] - forward_result['output_layer'][j])**2 
                                   for j in range(self.output_size))
            
            # Print error every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Total Error: {total_error}")
    
    def predict(self, input_vector):
        """
        Predict output for given input vector
        
        Args:
            input_vector (list): Input feature vector
        
        Returns:
            list: Predicted output
        """
        return self.forward_propagation(input_vector)['output_layer']

class TechTrendIRProcessor:
    def __init__(self):
        # Semantic dictionary for related terms
        self.semantic_dictionary = {
            "ai": ["machine learning", "artificial intelligence", "neural networks", "deep learning"],
            "blockchain": ["cryptocurrency", "decentralized", "web3", "smart contracts"],
            "cloud": ["cloud computing", "distributed systems", "serverless", "infrastructure"],
            "cybersecurity": ["network security", "data protection", "encryption", "privacy"],
            "startups": ["innovation", "tech companies", "entrepreneurship", "venture capital"]
        }
        
        # Articles database
        self.articles = [
            {
                "title": "The Rise of Generative AI in Enterprise Solutions",
                "content": "Generative AI is transforming enterprise software, offering unprecedented capabilities in automation, content creation, and decision support systems.",
                "keywords": ["ai", "machine learning", "enterprise"],
                "features": [0.7, 0.5, 0.6, 0.4]  # Extended feature vector
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
        
        # Initialize custom neural network
        self.nn = NeuralNetwork(input_size=4, hidden_size=5, output_size=1)
        
        # Prepare training data
        self._prepare_training_data()
    
    def _prepare_training_data(self):
        """
        Prepare training data for the neural network
        """
        # Extract features and create training labels
        X_train = [article['features'] for article in self.articles]
        
        # Create synthetic relevance labels
        def calculate_relevance(article):
            # More keywords match = higher relevance
            return [min(1.0, len(set(article['keywords'])) / 3.0)]
        
        y_train = [calculate_relevance(article) for article in self.articles]
        
        # Train the neural network
        self.nn.train(X_train, y_train, epochs=1000)
    
    def preprocess_text(self, query):
        """
        Preprocess query and extract keywords
        
        Args:
            query (str): Input query string
        
        Returns:
            list: Preprocessed keywords
        """
        # Simple preprocessing
        return [word.lower() for word in query.split()]
    
    def semantic_understanding(self, keywords):
        """
        Find semantic relationships for keywords
        
        Args:
            keywords (list): Input keywords
        
        Returns:
            dict: Semantic relationships
        """
        semantic_map = {}
        for word in keywords:
            related_terms = self.semantic_dictionary.get(word, [])
            semantic_map[word] = related_terms
        return semantic_map
    
    def query_expansion(self, keywords, semantic_map):
        """
        Expand query with related terms
        
        Args:
            keywords (list): Original keywords
            semantic_map (dict): Semantic relationships
        
        Returns:
            list: Expanded query terms
        """
        expanded_terms = set(keywords)
        for word in keywords:
            expanded_terms.update(semantic_map.get(word, []))
        return list(expanded_terms)
    
    def extract_features(self, expanded_terms):
        """
        Extract numerical features for neural network
        
        Args:
            expanded_terms (list): Expanded query terms
        
        Returns:
            list: Numerical feature vector
        """
        # Feature extraction logic
        return [
            len(expanded_terms) / 10.0,  # Term count
            len(set(expanded_terms)) / len(expanded_terms),  # Unique term ratio
            sum(len(term) for term in expanded_terms) / len(expanded_terms),  # Avg term length
            len([term for term in expanded_terms if any(t in term for t in self.semantic_dictionary)])  # Semantic match count
        ]
    
    def search_and_retrieve(self, expanded_terms):
        """
        Search and retrieve relevant articles
        
        Args:
            expanded_terms (list): Expanded query terms
        
        Returns:
            list: Matching articles with relevance scores
        """
        # Extract features for the query
        features = self.extract_features(expanded_terms)
        
        # Use neural network to score relevance
        matching_articles = []
        for article in self.articles:
            # Check semantic matching
            if any(term.lower() in ' '.join(article['keywords']).lower() or 
                   term.lower() in article.get('content', '').lower() 
                   for term in expanded_terms):
                # Predict relevance using neural network
                relevance_score = self.nn.predict(features)[0]
                
                matching_articles.append({
                    'article': article,
                    'relevance_score': relevance_score
                })
        
        # Sort by relevance score
        return sorted(matching_articles, key=lambda x: x['relevance_score'], reverse=True)
    
    def process_query(self, query):
        """
        Complete query processing pipeline
        
        Args:
            query (str): Input query string
        
        Returns:
            list: Ranked article titles with relevance scores
        """
        # 1. Preprocess text
        keywords = self.preprocess_text(query)
        
        # 2. Semantic understanding
        semantic_map = self.semantic_understanding(keywords)
        
        # 3. Query expansion
        expanded_terms = self.query_expansion(keywords, semantic_map)
        
        # 4. Search and retrieve
        matching_articles = self.search_and_retrieve(expanded_terms)
        
        # 5. Return ranked article titles
        return [f"{result['article']['title']} (Relevance: {result['relevance_score']:.2f})" 
                for result in matching_articles]

def main():
    # Create the Information Retrieval Processor
    ir_processor = TechTrendIRProcessor()
    
    # Interactive query processing
    while True:
        query = input('Enter your technology query (or "quit" to exit): ')
        
        if query.lower() == 'quit':
            break
        
        results = ir_processor.process_query(query)
        print("\nMatching Articles:")
        for result in results:
            print(f"- {result}")

if __name__ == "__main__":
    main()