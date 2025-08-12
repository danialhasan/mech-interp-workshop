"""
Flask Web Interface for Steering Vector Demo
HasanLabs - Mechanistic Interpretability Workshop
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from llama_3b_steered.steering_vectors import SteeredLlama, SteeringVector

app = Flask(__name__)
CORS(app)

# Global model instance (loaded once)
model = None
vectors = {}


def initialize_model():
    """Initialize the model and load steering vectors."""
    global model, vectors
    
    print("Initializing Llama 3B model...")
    model = SteeredLlama()
    
    # Load pre-computed vectors
    vector_dir = Path(__file__).parent.parent / "llama_3b_steered" / "vectors"
    
    if (vector_dir / "weeknd.pkl").exists():
        vectors['weeknd'] = SteeringVector.load(str(vector_dir / "weeknd.pkl"))
        print("Loaded Weeknd steering vector")
    
    if (vector_dir / "toronto.pkl").exists():
        vectors['toronto'] = SteeringVector.load(str(vector_dir / "toronto.pkl"))
        print("Loaded Toronto steering vector")
    
    if (vector_dir / "tabby_cats.pkl").exists():
        vectors['tabby_cats'] = SteeringVector.load(str(vector_dir / "tabby_cats.pkl"))
        print("Loaded Tabby Cats steering vector")
    
    print("Model and vectors loaded successfully!")


@app.route('/')
def index():
    """Main interface page."""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate text with optional steering."""
    global model
    
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    data = request.json
    prompt = data.get('prompt', '')
    steering_type = data.get('steering', 'none')
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    
    try:
        # Clear any existing steering
        model.clear_steering_vectors()
        
        # Apply requested steering
        if steering_type != 'none' and steering_type in vectors:
            model.add_steering_vector(vectors[steering_type])
            print(f"Applied {steering_type} steering")
        
        # Generate output
        output = model.generate(
            prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        # Extract just the generated part
        generated_text = output[len(prompt):].strip()
        
        return jsonify({
            'success': True,
            'prompt': prompt,
            'generated': generated_text,
            'steering': steering_type
        })
    
    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/compare', methods=['POST'])
def compare():
    """Generate both base and steered outputs for comparison."""
    global model
    
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    data = request.json
    prompt = data.get('prompt', '')
    steering_type = data.get('steering', 'weeknd')
    max_length = data.get('max_length', 100)
    
    try:
        # Generate base output
        model.clear_steering_vectors()
        base_output = model.generate(prompt, max_length=max_length)
        base_text = base_output[len(prompt):].strip()
        
        # Generate steered output
        if steering_type in vectors:
            model.clear_steering_vectors()
            model.add_steering_vector(vectors[steering_type])
            steered_output = model.generate(prompt, max_length=max_length)
            steered_text = steered_output[len(prompt):].strip()
        else:
            steered_text = "Invalid steering type"
        
        return jsonify({
            'success': True,
            'prompt': prompt,
            'base': base_text,
            'steered': steered_text,
            'steering_type': steering_type
        })
    
    except Exception as e:
        print(f"Comparison error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/examples', methods=['GET'])
def get_examples():
    """Return pre-defined example prompts."""
    examples = [
        "The future of technology is",
        "Climate change is affecting",
        "The best way to learn programming is",
        "In the stock market today,",
        "The recipe for happiness includes",
        "Artificial intelligence will",
        "The meaning of life is",
        "Tomorrow's weather will be"
    ]
    return jsonify(examples)


if __name__ == '__main__':
    print("Starting HasanLabs Steering Demo Server...")
    print("Initializing model (this may take a minute)...")
    initialize_model()
    print("\n" + "="*50)
    print("Server ready! Visit http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000, host='0.0.0.0')