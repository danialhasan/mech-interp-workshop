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
    
    print("Initializing TinyLlama model...")
    model = SteeredLlama(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Load pre-computed vectors - now loading multiple layers per persona
    vector_dir = Path(__file__).parent.parent / "llama_3b_steered" / "vectors"
    
    strength_override = os.environ.get('STRENGTH')
    strength_value = None
    if strength_override is not None:
        try:
            strength_value = float(strength_override)
            print(f"Overriding vector strength to {strength_value}")
        except ValueError:
            print(f"Invalid STRENGTH value: {strength_override}; ignoring override")

    # Load multi-layer vectors for each persona
    for persona in ['weeknd', 'toronto', 'tabby_cats']:
        vectors[persona] = []
        loaded_layers = []
        
        # Try to load vectors for layers 10-15
        for layer in range(10, 16):
            vec_path = vector_dir / f"{persona}_L{layer}.pkl"
            if vec_path.exists():
                vec = SteeringVector.load(str(vec_path), device=model.device)
                if strength_value is not None:
                    vec.strength = strength_value
                vectors[persona].append(vec)
                loaded_layers.append(layer)
        
        if loaded_layers:
            print(f"Loaded {persona} steering vectors for layers: {loaded_layers}")
        else:
            # Fallback to single vector if exists
            single_path = vector_dir / f"{persona}.pkl"
            if single_path.exists():
                vec = SteeringVector.load(str(single_path), device=model.device)
                if strength_value is not None:
                    vec.strength = strength_value
                vectors[persona] = [vec]
                print(f"Loaded single {persona} steering vector (fallback)")
    
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
    strength_override = data.get('strength', None)
    
    try:
        # Clear any existing steering
        model.clear_steering_vectors()
        
        # Apply requested steering (now supports multiple layers)
        if steering_type != 'none' and steering_type in vectors:
            vec_list = vectors[steering_type]
            applied_layers = []
            for vec in vec_list:
                if strength_override is not None:
                    try:
                        s = float(strength_override)
                        temp_vec = SteeringVector(vec.vector, vec.layer_idx, s)
                    except Exception:
                        temp_vec = vec
                else:
                    temp_vec = vec
                model.add_steering_vector(temp_vec)
                applied_layers.append(vec.layer_idx)
            print(f"Applied {steering_type} steering at layers: {applied_layers}")
        
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
    strength_override = data.get('strength', None)
    
    try:
        # Generate base output
        model.clear_steering_vectors()
        base_output = model.generate(prompt, max_length=max_length)
        base_text = base_output[len(prompt):].strip()
        
        # Generate steered output (now supports multiple layers)
        if steering_type in vectors:
            model.clear_steering_vectors()
            vec_list = vectors[steering_type]
            applied_layers = []
            for vec in vec_list:
                if strength_override is not None:
                    try:
                        s = float(strength_override)
                        temp_vec = SteeringVector(vec.vector, vec.layer_idx, s)
                    except Exception:
                        temp_vec = vec
                else:
                    temp_vec = vec
                model.add_steering_vector(temp_vec)
                applied_layers.append(vec.layer_idx)
            print(f"Applied {steering_type} steering at layers: {applied_layers}")
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
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '5050'))
    print(f"Server ready! Visit http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=False, port=port, host=host)