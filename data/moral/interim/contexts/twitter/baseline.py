class MoralValue:
    def __init__(self):
        pass

class CareOrHarm(MoralValue):
    """Prescriptive moral values such as caring for others, generosity and compassion and moral values prohibiting actions that harm others.

    This foundation is related to our long evolution as mammals with attachment systems and an ability to feel (and dislike) the pain of others. It underlies virtues of kindness, gentleness, and nurturance.
    """
    pass
    
class FairnessOrCheating(MoralValue):
    """Prescriptive moral values such as fairness, justice, and reciprocity and moral values prohibiting cheating.

    This foundation is related to the evolutionary process of reciprocal altruism. It generates ideas of justice, rights, and autonomy.
    """
    pass

class LoyaltyOrBetrayal(MoralValue):
    """Prescriptive moral values associated with group affiliation and solidarity and moral values prohibiting betrayal of one’s group.

    This foundation is related to our long history as tribal creatures able to form shifting coalitions. It underlies virtues of patriotism and self-sacrifice for the group. It is active anytime people feel that it’s “one for all, and all for one.”
    """
    pass
    
class AuthorityOrSubversion(MoralValue):
    """Prescriptive moral values associated with fulfilling social roles and submitting to authority and moral values prohibiting rebellion against authority.

    This foundation was shaped by our long primate history of hierarchical social interactions. It underlies virtues of leadership and followership including deference to legitimate authority and respect for traditions.
    """
    pass

class PurityOrDegradation(MoralValue):
    """Prescriptive moral values associated with the sacred and holy and moral values prohibiting violating the sacred.

    This foundation was shaped by the psychology of disgust and contamination. It underlies religious notions of striving to live in an elevated, less carnal, more noble way. It underlies the widespread idea that the body is a temple that can be desecrated by immoral activities and contaminants (an idea not unique to religious traditions).
    """
    pass
    
class MoralSentimentPrediction:
    def __init__(
        self,
        care_or_harm: CareOrHarm | None = None,
        fairness_or_cheating: FairnessOrCheating | None = None,
        loyalty_or_betrayal: LoyaltyOrBetrayal | None = None,
        authority_or_subversion: AuthorityOrSubversion | None = None,
        purity_or_degradation: PurityOrDegradation | None = None
    ):
        self.care_or_harm = care_or_harm
        self.fairness_or_cheating = fairness_or_cheating
        self.loyalty_or_betrayal = loyalty_or_betrayal
        self.authority_or_subversion = authority_or_subversion
        self.purity_or_degradation = purity_or_degradation
