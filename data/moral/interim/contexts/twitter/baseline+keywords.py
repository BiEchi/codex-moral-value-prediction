class MoralValue:
    def __init__(self, sentiment: bool):
        self.sentiment = sentiment

class CareOrHarm(MoralValue):
    """Prescriptive moral values such as caring for others, generosity and compassion and moral values prohibiting actions that harm others.

    This foundation is related to our long evolution as mammals with attachment systems and an ability to feel (and dislike) the pain of others. It underlies virtues of kindness, gentleness, and nurturance.

    When self.sentiment == True, it represents moral values such as caring for others, generosity and compassion. 
    Otherwise it represent actions that harm others.

    Keywords:
    Care (self.sentiment == True): save, defend, protect, compassion
    Harm (self.sentiment == False): harm, war, kill, suffer
    """
    pass
    
class FairnessOrCheating(MoralValue):
    """Prescriptive moral values such as fairness, justice, and reciprocity and moral values prohibiting cheating.

    This foundation is related to the evolutionary process of reciprocal altruism. It generates ideas of justice, rights, and autonomy.

    When self.sentiment == True, it represents fairness.
    Otherwise it represent cheating.

    Keywords:
    Fairness (self.sentiment == True): fair, equal, justice, honesty
    Cheating (self.sentiment == False): unfair, unequal, unjust, dishonest
    """
    pass

class LoyaltyOrBetrayal(MoralValue):
    """Prescriptive moral values associated with group affiliation and solidarity and moral values prohibiting betrayal of one’s group.

    This foundation is related to our long history as tribal creatures able to form shifting coalitions. It underlies virtues of patriotism and self-sacrifice for the group. It is active anytime people feel that it’s “one for all, and all for one.”

    When self.sentiment == True, it represents loyalty.
    Otherwise it represent betrayal of one’s group.

    Keywords:
    Loyalty (self.sentiment == True): solidarity, nation, family, support
    Betrayal (self.sentiment == False): betray (in-group), abandon, rebel (against in-group)
    """
    pass
    
class AuthorityOrSubversion(MoralValue):
    """Prescriptive moral values associated with fulfilling social roles and submitting to authority and moral values prohibiting rebellion against authority.

    This foundation was shaped by our long primate history of hierarchical social interactions. It underlies virtues of leadership and followership including deference to legitimate authority and respect for traditions.

    When self.sentiment == True, it represents fulfilling social roles and submitting to authority.
    Otherwise it represent rebellion against authority.

    Keywords:
    Authority (self.sentiment == True): duty, law, obligation, order
    Subversion (self.sentiment == False): rebel (against authority), chaos, disorder, betray (your role)
    """
    pass

class PurityOrDegradation(MoralValue):
    """Prescriptive moral values associated with the sacred and holy and moral values prohibiting violating the sacred.

    This foundation was shaped by the psychology of disgust and contamination. It underlies religious notions of striving to live in an elevated, less carnal, more noble way. It underlies the widespread idea that the body is a temple that can be desecrated by immoral activities and contaminants (an idea not unique to religious traditions).

    When self.sentiment == True, it represents moral values associated with the sacred and holy.
    Otherwise it represent violating the sacred.

    Keywords:
    Purity (self.sentiment == True): sacred, preserve, pure, serenity
    Degradation (self.sentiment == False): dirty, repulsive, disgusting, revolting
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
