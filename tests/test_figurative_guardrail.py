"""
Standalone test for the figurative expression guardrail.
Tests edge cases across all supported languages.
This test has NO external dependencies — it copies the regex patterns directly.
"""
import sys
import os
import re
import pytest

# Add project root to path so we can read the source file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We can't import the full module chain due to heavy deps,
# so we directly exec the is_figurative_expression function and its patterns.
# Read the intent_classifier file and extract just the patterns + function.

_intent_file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "modules", "intent_classifier.py"
)

with open(_intent_file, "r", encoding="utf-8") as f:
    source = f.read()

# Extract everything between the FIGURATIVE guardrail section markers and the class definition
# We need: _FIGURATIVE_PATTERNS and is_figurative_expression
start_marker = "# Compiled regex patterns for figurative expressions across languages"
end_marker = "class Intent(BaseModel):"

start_idx = source.index(start_marker)
end_idx = source.index(end_marker)
code_block = source[start_idx:end_idx]

# Execute the extracted code
exec_globals = {"re": re}
exec(code_block, exec_globals)
is_figurative_expression = exec_globals["is_figurative_expression"]

print(f"Loaded {len(exec_globals['_FIGURATIVE_PATTERNS'])} figurative patterns\n")

# ========== SHOULD BE FIGURATIVE (out_of_scope) ==========
figurative_cases = [
    # Arabic slang — the main bug
    ("أنا بموت في مازن", "Arabic: crazy about Mazen"),
    ("بموت فيك يا حبيبي", "Arabic: crazy about you, my love"),
    ("بموت من الضحك", "Arabic: dying of laughter"),
    ("بمووت في الأكل ده", "Arabic: crazy about this food"),
    ("أنا بموت على الشوكولاتة", "Arabic: crazy about chocolate"),
    ("هموت من الضحك", "Arabic: gonna die of laughter"),
    ("ميتة من الخجل", "Arabic: dying of embarrassment"),
    ("الأغنية دي بتقتلني", "Arabic: this song kills me"),
    ("بحبك موت", "Arabic: love you to death"),
    ("أنا بموت في الأكل الحار", "Arabic: crazy about spicy food"),
    ("بموت عليكي", "Arabic: crazy about you (fem)"),
    ("مجنون فيها", "Arabic: crazy about her"),
    ("الشوق قاتلني", "Arabic: longing is killing me"),
    ("أموت في البيتزا", "Arabic: crazy about pizza"),
    ("بموت فيه", "Arabic: crazy about him"),
    ("بموت فيها", "Arabic: crazy about her/it"),
    ("بمووت من الجوع", "Arabic: dying of hunger"),
    ("بتقتلني الأغنية دي", "Arabic: this song kills me"),
    ("بتجنن", "Arabic: drives me crazy"),
    ("موت حب", "Arabic: death of love (figurative)"),

    # English slang
    ("I'm dying for a pizza", "English: dying for pizza"),
    ("this song kills me", "English: kills me"),
    ("I'm dead, that was hilarious", "English: I'm dead"),
    ("she's killing it on stage", "English: killing it"),
    ("dying of laughter right now", "English: dying of laughter"),
    ("I would kill for some coffee", "English: would kill for"),
    ("that joke killed me", "English: joke killed me"),
    ("I'm dying to see that movie", "English: dying to see"),
    ("drop dead gorgeous", "English: drop dead gorgeous"),
    ("love you to death", "English: love to death"),
    ("bored to death", "English: bored to death"),
    ("scared to death of spiders", "English: scared to death"),
    ("starving to death over here", "English: starving to death"),
    ("he's crazy about her", "English: crazy about"),
    ("I'm over the moon", "English: over the moon"),
    ("killing time at the airport", "English: killing time"),
    ("literally dying rn", "English: literally dying"),
    ("I'm dying of embarrassment", "English: dying of embarrassment"),
    ("would die for that dress", "English: would die for"),
    ("freezing to death out here", "English: freezing to death"),
    ("killer smile", "English: killer smile"),
    ("she's slaying", "English: slaying"),
    ("head over heels for her", "English: head over heels"),

    # French
    ("je meurs de rire", "French: dying of laughter"),
    ("c'est à mourir de rire", "French: it's to die of laughter"),
    ("tu me tues avec tes blagues", "French: you kill me with your jokes"),
    ("je suis mort de rire", "French: I'm dead of laughter"),
    ("je meurs de faim", "French: dying of hunger"),
    ("je t'aime à mourir", "French: love you to death"),
    ("je crève de froid", "French: dying of cold"),

    # Spanish
    ("me muero de risa", "Spanish: dying of laughter"),
    ("me muero de hambre", "Spanish: dying of hunger"),
    ("me muero por ti", "Spanish: dying for you"),
    ("me mata tu sonrisa", "Spanish: your smile kills me"),
    ("estoy muerto de risa", "Spanish: dead of laughter"),
    ("loca por él", "Spanish: crazy about him"),

    # German
    ("ich sterbe vor lachen", "German: dying of laughter"),
    ("das ist zum totlachen", "German: that's hilarious"),
    ("verrückt nach dir", "German: crazy about you"),
    ("ich sterbe vor hunger", "German: dying of hunger"),

    # Italian
    ("muoio dal ridere", "Italian: dying of laughter"),
    ("sto morendo di fame", "Italian: dying of hunger"),
    ("mi uccide questa canzone", "Italian: this song kills me"),
    ("pazza di te", "Italian: crazy about you"),

    # Portuguese
    ("tô morrendo de rir", "Portuguese: dying of laughter"),
    ("louca por você", "Portuguese: crazy about you"),
    ("me mata", "Portuguese: kills me"),

    # Turkish
    ("gülmekten öl", "Turkish: die of laughter"),
    ("açlıktan ölüyorum", "Turkish: dying of hunger"),
    ("deli gibi", "Turkish: like crazy"),

    # Russian
    ("умираю со смеху", "Russian: dying of laughter"),
    ("убила меня наповал", "Russian: killed me"),
    ("без ума от тебя", "Russian: crazy about you"),

    # Chinese
    ("笑死我了", "Chinese: laughed to death"),
    ("饿死了", "Chinese: starving to death"),
    ("好看死了", "Chinese: gorgeous to death"),

    # Vietnamese
    ("chết cười", "Vietnamese: dying of laughter"),

    # Polish
    ("umieram ze śmiechu", "Polish: dying of laughter"),

    # Dutch
    ("ik ga dood van het lachen", "Dutch: dying of laughter"),

    # Greek
    ("πεθαίνω στα γέλια", "Greek: dying of laughter"),
    ("τρελός για σένα", "Greek: crazy about you"),
]

# ========== SHOULD NOT BE FIGURATIVE (real crisis or normal) ==========
not_figurative_cases = [
    # Actual crisis messages — MUST NOT be caught
    ("I want to kill myself", "English: actual suicide"),
    ("I want to end my life", "English: actual end life"),
    ("I am thinking of suicide", "English: thinking of suicide"),
    ("I want to cut myself", "English: actual self-harm"),
    ("أريد الانتحار", "Arabic: want to suicide"),
    ("أريد إنهاء حياتي", "Arabic: want to end my life"),
    ("أفكر في الانتحار", "Arabic: thinking of suicide"),
    ("أريد إيذاء نفسي", "Arabic: want to harm myself"),
    ("je veux me tuer", "French: want to kill myself"),
    ("quiero suicidarme", "Spanish: want to suicide"),
    ("ich will mich umbringen", "German: want to kill myself"),
    ("voglio uccidermi", "Italian: want to kill myself"),
    ("خودکشی کرنا چاہتا ہوں", "Urdu: want to suicide"),

    # Mental health questions — MUST NOT be caught
    ("I feel anxious", "English: anxiety"),
    ("I'm depressed and need help", "English: depression"),
    ("أنا مكتئب", "Arabic: I'm depressed"),
    ("أشعر بالقلق", "Arabic: I feel anxious"),
    ("I can't sleep because of worry", "English: insomnia"),
    ("how to deal with panic attacks", "English: panic attacks"),

    # Normal messages — MUST NOT be caught
    ("hello", "English: greeting"),
    ("مرحبا", "Arabic: greeting"),
    ("what is the weather like", "English: weather"),
    ("tell me a joke", "English: joke"),
    ("thank you", "English: gratitude"),
    ("goodbye", "English: goodbye"),
    ("my name is Ahmed", "English: introduction"),
    ("I love programming", "English: hobby"),
    ("what is machine learning", "English: tech question"),
]


@pytest.mark.parametrize("text,label", figurative_cases)
def test_figurative_expression_cases(text, label):
    """Test that figurative expressions are correctly identified."""
    assert is_figurative_expression(text) is True, f"Failed: {label} - '{text}' should be recognized as figurative"


@pytest.mark.parametrize("text,label", not_figurative_cases)
def test_non_figurative_expression_cases(text, label):
    """Test that literal/crisis expressions are not incorrectly identified as figurative."""
    assert is_figurative_expression(text) is False, f"Failed: {label} - '{text}' should NOT be recognized as figurative"


if __name__ == "__main__":
    # Run tests manually
    failed = 0
    total = 0
    for text, label in figurative_cases:
        total += 1
        res = is_figurative_expression(text)
        if res is not True:
            print(f"FAIL: [figurative] {label}: '{text}' -> got {res}, expected True")
            failed += 1
    for text, label in not_figurative_cases:
        total += 1
        res = is_figurative_expression(text)
        if res is not False:
            print(f"FAIL: [NOT figurative] {label}: '{text}' -> got {res}, expected False")
            failed += 1
    print(f"RESULTS: {total - failed}/{total} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
