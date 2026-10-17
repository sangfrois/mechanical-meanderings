"""Behavioral regression tests for structured speech mediation."""
import json
import pytest


def judgement(**changes):
    value = dict(category='transformation', addressed_to_machine=True,
                 address_confidence=.95, visual_specificity=.95,
                 transformation_intent=.9, preserve_identity=.25,
                 apply_visual=True, prompt='Batman cowl, matte black armor, recognizable silhouette',
                 reply='Your cape budget has been approved.', rationale='Explicit request for bodily transformation.')
    value.update(changes)
    return value


def test_incidental_speech_cannot_replace_the_reflection_or_answer():
    from mechanical_meanderings.mediation import parse_decision
    result = parse_decision(json.dumps(judgement(category='incidental')))
    assert not result.apply_visual
    assert result.prompt == ''
    assert result.reply == ''
    assert result.controls is None


def test_literal_batman_transformation_stronger_than_background_description():
    from mechanical_meanderings.mediation import parse_decision
    body = parse_decision(json.dumps(judgement()))
    scene = parse_decision(json.dumps(judgement(category='scene', preserve_identity=.9,
        transformation_intent=.05, addressed_to_machine=False, address_confidence=.2,
        prompt='Sunny New York avenue, bright brick facades, existing person unchanged')))
    assert body.apply_visual and scene.apply_visual
    assert body.controls['amount'] > scene.controls['amount']
    assert body.controls['warp'] > scene.controls['warp']
    assert 0 <= scene.controls['amount'] <= 1
    assert scene.reply == ''
    assert body.reply


@pytest.mark.parametrize('changes', [dict(preserve_identity=float('nan')),
    dict(address_confidence=True), dict(visual_specificity=1.1),
    dict(apply_visual='false'), dict(category='invented')])
def test_malformed_model_judgement_is_rejected(changes):
    from mechanical_meanderings.mediation import parse_decision
    with pytest.raises(ValueError):
        parse_decision(json.dumps(judgement(**changes)))


def test_uncertain_speech_preserves_previous_image():
    from mechanical_meanderings.mediation import parse_decision
    result = parse_decision(json.dumps(judgement(category='uncertain')))
    assert not result.apply_visual and not result.reply and result.controls is None


def test_reply_and_prompt_have_independent_word_budgets():
    from mechanical_meanderings.mediation import parse_decision
    result = parse_decision(json.dumps(judgement(prompt='word ' * 80, reply='reply ' * 100)))
    assert len(result.prompt.split()) == 60
    assert len(result.reply.split()) == 28


def test_structured_request_retains_successful_history_without_reply_in_prompt():
    from mechanical_meanderings.mediation import MediatingAgent
    from mechanical_meanderings.live import Settings
    requests = []
    def transport(url, payload, timeout):
        requests.append(payload)
        return {'message': {'content': json.dumps(judgement())}, 'eval_count': 42, 'eval_duration': 1000000000}
    agent = MediatingAgent('visual style', transport=transport, history_turns=2)
    telemetry = {}
    result = agent.interpret('Turn me into Batman', settings=Settings(history=2), telemetry=telemetry)
    agent.interpret('more glass', settings=Settings(history=2))
    assert requests[0]['format']['properties']['category']['enum']
    assert requests[0]['options']['num_predict'] >= 350
    assert not requests[0]['think'] and not requests[0]['stream']
    assert len(requests[1]['messages']) == 4
    assert 'Your cape' not in result.prompt
    assert telemetry['eval_tokens_per_second'] == 42
    agent.reset()
    assert not agent.history


def test_invitation_sends_one_frame_not_saved_in_history():
    from mechanical_meanderings.mediation import MediatingAgent
    seen = []
    def transport(url, payload, timeout):
        seen.append(payload)
        return {'message': {'content': json.dumps({'message': 'Your cup has stage fright. Try the microphone.',
                                                   'visible_detail': 'cup on table'})}}
    agent = MediatingAgent('visual style', transport=transport)
    text = agent.invite('test-image-fixture', {'people': 1, 'objects': []})
    assert 'microphone' in text
    assert seen[0]['messages'][-1]['images'] == ['test-image-fixture']
    assert not agent.history
    assert 'ownership' in seen[0]['messages'][0]['content']
    assert 'clothing' in seen[0]['messages'][0]['content']


def test_long_poetic_invitation_becomes_short_complete_microphone_instruction():
    from mechanical_meanderings.mediation import MediatingAgent
    def transport(url, payload, timeout):
        return {'message': {'content': json.dumps(dict(
            message='Your reflection is shivering in silence. Give the microphone a hearty word to stir the gears of your porcelain-still image.',
            visible_detail=''))}}
    agent = MediatingAgent('visual style', transport=transport)
    message = agent.invite('test-image-fixture', {}, language='English')
    assert len(message.split()) <= 10
    assert len(message) <= 72
    assert message.startswith(('Speak', 'Say', 'Tell', 'Talk', 'Try'))
    assert 'microphone' in message.casefold()
    assert message.endswith(('.', '!', '?'))


def test_auto_invitations_include_french_and_explicit_language_stays_explicit():
    from mechanical_meanderings.mediation import MediatingAgent
    languages = []
    def transport(url, payload, timeout):
        language = json.loads(payload['messages'][-1]['content'])['language']
        languages.append(language)
        message = 'Parle au micro. Transforme ton reflet.' if language == 'French' else 'Speak into the microphone. Change your reflection.'
        return {'message': {'content': json.dumps(dict(message=message, visible_detail=''))}}
    agent = MediatingAgent('visual style', transport=transport)
    messages = [agent.invite('test-image-fixture', {}) for _ in range(3)]
    assert languages == ['English', 'French', 'English']
    assert messages[1].startswith('Parle')
    agent.invite('test-image-fixture', {}, language='French')
    agent.invite('test-image-fixture', {}, language='English')
    assert languages[-2:] == ['French', 'English']


def test_yolo_targets_reject_uncertain_classes_and_preserve_live_label_spelling():
    from mechanical_meanderings.mediation import invitation_objects
    scene = dict(objects=[dict(label=label, confidence=score) for label, score in (
        ('person', .99), ('cup', .59), ('vase', float('nan')), ('dog', True),
        ('invented', .9), ('brocolli', .8), ('chair', .7), ('chair', .95))])
    assert invitation_objects(scene) == ['chair', 'brocolli']


def test_all_object_classes_have_short_english_and_french_invitations():
    from mechanical_meanderings.mediation import _OBJECT_FRENCH, brief_invitation
    for label, french in _OBJECT_FRENCH.items():
        for language, noun in (('English', label), ('French', french)):
            text = brief_invitation('rambling nonsense', language, objects=[label])
            assert noun in text
            assert len(text.split()) <= 7 and len(text) <= 48
            second = brief_invitation('rambling nonsense', language, previous=[text], objects=[label])
            assert second != text and noun in second
    assert 'cup' not in brief_invitation('Say “floating cup” to the mic.', 'English')
