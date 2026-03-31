from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # allows the HTML frontend to call this API

# ── Load model and scaler ──────────────────────────────────────
rf     = joblib.load('personality_model.pkl')
scaler = joblib.load('personality_scaler.pkl')

trait_cols = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

# ── Personality type mapping ───────────────────────────────────
def get_personality_type(pred):
    profile = {trait: pred[i] for i, trait in enumerate(trait_cols)}
    o = profile['openness']
    c = profile['conscientiousness']
    e = profile['extraversion']
    a = profile['agreeableness']
    n = profile['neuroticism']

    if e=='High' and a=='High' and n=='Low':
        return 'The Leader',     'Charismatic, warm, and emotionally stable. Natural at bringing people together and inspiring those around them.'
    if o=='High' and e=='High' and a=='High':
        return 'The Entertainer','Creative, sociable, and fun-loving. Thrives in social settings and brings energy wherever they go.'
    if c=='High' and o=='High' and n=='Low':
        return 'The Achiever',   'Disciplined, curious, and focused. Highly goal-oriented with a strong drive to learn and excel.'
    if a=='High' and n=='Low' and e=='Low':
        return 'The Caregiver',  'Empathetic, calm, and deeply supportive. Puts others first and brings stability to every relationship.'
    if o=='High' and e=='Low' and c=='High':
        return 'The Thinker',    'Intellectual, introspective, and methodical. Prefers the world of ideas and thrives in deep focus.'
    if n=='High' and e=='Low' and a=='Low':
        return 'The Lone Wolf',  'Independent, intense, and private. Prefers depth over breadth and carves their own path.'
    if c=='High' and a=='High' and e=='Low':
        return 'The Defender',   'Reliable, kind, and hardworking. Quietly dedicated to doing what is right and caring for others.'
    if o=='Low' and c=='Low' and e=='High':
        return 'The Adventurer', 'Spontaneous, energetic, and thrill-seeking. Lives fully in the present moment.'
    if o=='High' and a=='Low' and e=='High':
        return 'The Debater',    'Bold, curious, and argumentative. Loves challenging ideas and never backs down from a discussion.'
    return 'The Balanced', 'A well-rounded and adaptable personality. Comfortable across many different situations and social contexts.'


# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({'status': 'OCEAN API running'})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Expect: { "responses": [1, 3, 4, 2, ...] }  (120 integers, 1-5)
        responses = data.get('responses', [])

        if len(responses) != 120:
            return jsonify({'error': f'Expected 120 responses, got {len(responses)}'}), 400

        if not all(isinstance(r, (int, float)) and 1 <= r <= 5 for r in responses):
            return jsonify({'error': 'All responses must be integers between 1 and 5'}), 400

        # Scale and predict
        user_input = scaler.transform(np.array([responses]))
        pred       = rf.predict(user_input)[0]   # e.g. ['Low', 'High', 'Medium', 'Low', 'High']

        # Map predictions to dict
        traits = {trait: pred[i] for i, trait in enumerate(trait_cols)}

        # Get personality type
        ptype, pdesc = get_personality_type(pred)

        return jsonify({
            'traits':           traits,
            'personality_type': ptype,
            'description':      pdesc
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
