"""Templates for the Spelling Standardisation task."""

from ..data_models import PromptConfig
from ..languages import ICELANDIC

SPELL_TEMPLATES = {
    ICELANDIC: PromptConfig(
        default_prompt_prefix="Eftirfarandi eru setningar ásamt útgáfum þeirra sem "
        "eru í samræmi við íslenskar ritreglur.",
        default_prompt_template="Setning: {text}\nStöðluð setning: {target_text}",
        default_instruction_prompt="Setning: {text}\n\nHugsanlega er villa í "
        "stafsetningu eða greinarmerkjasetningu í setningunni hér að ofan, en ekki er "
        "víst að svo sé. Skrifaðu setninguna í samræmi við íslenskar ritreglur, "
        "leiðrétta ef þörf krefur, og ekkert annað.",
        default_prompt_label_mapping=dict(),
    )
}
