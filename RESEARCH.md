# Visual interpreter: research and design rationale

This is a researched prompt design, not model fine-tuning, a visual-perception system, or a guarantee that a 12B model has mastered every named tradition.

## Installation and interaction

- Kairos-Hive, Mechanical Meanderings: https://kairos-hive.org/projects/mechanical-meanderings
  Describes a poetic mirror translating speech and physical presence into transformed live imagery, not a predetermined aesthetic. Interpretation, visitor negotiation, and unexpected translations matter. Design decision: preserve speaker intent, permit a bounded visual metaphor, and never automatically impose neon/cinematic/photorealistic imagery.
- Gordon Pask, A comment, a case history and a plan: https://www.pangaro.com/pask/Pask%20Cybernetic%20Serendipity%20Musicolour%20and%20Colloquy%20of%20Mobiles.pdf
  Describes novelty, learning to control an environment, and reactive/adaptive aesthetic environments. This informed the distinction between meaningful surprise and arbitrary decorative drift.
- ZKM, The Colloquy of Mobiles: https://zkm.de/en/artworks/the-colloquy-of-mobiles
  Describes communicating mobiles and visitor intervention through light and mirrors. Design decision: make contextual negotiation explicit, rather than making the interpreter an all-knowing chatbot. The text agent cannot see the camera.

## Material and cultural specificity

- The Metropolitan Museum of Art, Geometric Patterns in Islamic Art: https://www.metmuseum.org/essays/geometric-patterns-in-islamic-art
  Explains combinations, duplication, interlacing, symmetry, and potentially infinite growth from geometric units. Use actual formal operations rather than the vague label 'exotic ornament'.
- The Metropolitan Museum of Art, Woodblock Prints in the Ukiyo-e Style: https://www.metmuseum.org/essays/woodblock-prints-in-the-ukiyo-e-style
  Describes relief carving, ink/color printing, and historical subjects. Use process and pictorial qualities rather than reducing the tradition to one famous artist.
- The Metropolitan Museum of Art, Andean Textiles: https://www.metmuseum.org/essays/andean-textiles
  Describes weaving, dyeing, knotting, plaiting, identity, ancestral belief, and intercultural developments. Material knowledge can guide visual language without inventing a sacred meaning for a participant's image.
- The Metropolitan Museum of Art, How to Read African Textiles: https://www.metmuseum.org/met-publications/how-to-read-african-textiles
  Emphasizes rich and varied textile traditions, not one continental aesthetic. The prompt explicitly rejects interchangeable cultural decoration.

The additional repertoire in system.txt is a curatorial design proposal, not a claim that the above sources document every listed medium. It broadens possibilities across craft, popular imagery, scientific visualization, and digital media. No style is mandatory; explicit visitor choices outrank this repertoire.

## Context contract

Retain a bounded window of prior user/assistant pairs, not an ever-growing transcript. The latest correction wins. Assistant imagery is fallible interpretation and must not acquire the authority of the speaker's words. A new subject need not carry older motifs, and channel separation is not speaker identification. Physical visitor changes at the same microphone require a manual reset, an inactivity boundary, or interpretation of the spoken shift; no speaker recognition is claimed.

The word limit is both an instruction and a runtime guard. A 20-word cap is measured by whitespace-separated tokens, not Ollama tokens. Truncation enforces length but cannot guarantee semantic quality. Validate the actual generated images artistically before performance use.
