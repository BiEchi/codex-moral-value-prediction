class MoralValue:
    def __init__(self):
        pass
    
class AuthorityOrSubversion(MoralValue):
    """Prescriptive moral values such as respecting authority and following rules, and moral values promoting subversion of authority and breaking rules.
    
    This foundation is related to our need for social order and hierarchy, as well as our natural tendency to challenge authority. It underlies virtues such as obedience and defiance.

    If you think that the sentence is about authority or subversion, then you should instantiate the class with the argument `authority_or_subversion`.
    
    Note that we don't decide whether the sentiment is positive or negative. We just decide whether the sentence is about authority (or subversion).
    
    Keywords: 
    (positive): duty, law, obligation, order
    (negative): rebel (against authority), chaos, disorder, betray (your role)
    """
    pass
    
class CareOrHarm(MoralValue):
    """Prescriptive moral values such as caring for others, generosity and compassion and moral values prohibiting actions that harm others.

    This foundation is related to our long evolution as mammals with attachment systems and an ability to feel (and dislike) the pain of others. It underlies virtues of kindness, gentleness, and nurturance.

    If you think that the sentence is about care or harm, then you should instantiate the class with the argument `care_or_harm`.
    
    Note that we don't decide whether the sentiment is positive or negative. We just decide whether the sentence is about care (or harm).
    
    Keywords: 
    (positive): save, defend, protect, compassion, 
    (negative): harm, war, kill, suffer
    """
    pass

class EqualityOrInequality(MoralValue):
    """Prescriptive moral values such as equal treatment and fairness, and moral values prohibiting actions that result in unequal treatment or unfairness.
    
    This foundation is related to our long history as social animals with a strong desire for fairness and justice. It underlies virtues of justice, equality, and impartiality.

    If you think that the sentence is about equality or inequality, then you should instantiate the class with the argument `equality_or_inequality`.
    
    Note that we don't decide whether the sentiment is positive or negative. We just decide whether the sentence is about equality (or inequality).
    
    Keywords:
    (positive): fairness, justice, equality, impartiality
    (negative): inequality, unfairness, injustice, partiality
    """
    pass
    
class ProportionalityOrDisproportionality(MoralValue):
    """Prescriptive moral values associated with proportionality and fairness, and moral values prohibiting actions that are disproportionate or unfair to others.

    This foundation is related to our sense of fairness and justice, and our ability to weigh and compare the relative value or harm of different actions. It underlies virtues of justice, fairness, and equality.

    If you think that the sentence is about proportionality or disproportionality, then you should instantiate the class with the argument `proportionality_or_disproportionality`.
    
    Note that we don't decide whether the sentiment is positive or negative. We just decide whether the sentence is about proportionality (or disproportionality).
    
    Keywords:
    (positive): fair, equal, just, proportional
    (negative): disproportionate, unfair, unjust, unequal
    """
    pass

class LoyaltyOrBetrayal(MoralValue):
    """Prescriptive moral values associated with group affiliation and solidarity and moral values prohibiting betrayal of one’s group.

    This foundation is related to our long history as tribal creatures able to form shifting coalitions. It underlies virtues of patriotism and self-sacrifice for the group. It is active anytime people feel that it’s “one for all, and all for one.”

    If you think that the sentence is about loyalty or betrayal, then you should instantiate the class with the argument `loyalty_or_betrayal`.
    
    Note that we don't decide whether the sentiment is positive or negative. We just decide whether the sentence is about loyalty (or betrayal).
    
    Keywords:
    (positive): solidarity, nation, family, support
    (negative): betray (in-group), abandon, rebel (against in-group)
    """
    pass

class PurityOrDegradation(MoralValue):
    """Prescriptive moral values associated with the sacred and holy and moral values prohibiting violating the sacred.

    This foundation was shaped by the psychology of disgust and contamination. It underlies religious notions of striving to live in an elevated, less carnal, more noble way. It underlies the widespread idea that the body is a temple that can be desecrated by immoral activities and contaminants (an idea not unique to religious traditions).

    If you think that the sentence is about purity or degradation, then you should instantiate the class with the argument `purity_or_degradation`.
    
    Note that we don't decide whether the sentiment is positive or negative. We just decide whether the sentence is about purity (or degradation).
    
    Keywords:
    (positive): sacred, preserve, pure, serenity
    (negative): dirty, repulsive, disgusting, revolting
    """
    pass
    
class MoralSentimentPrediction:
    def __init__(
        self,
        authority_or_subversion: AuthorityOrSubversion | None = None,
        care_or_harm: CareOrHarm | None = None,
        equality_or_inequality: EqualityOrInequality | None = None,
        proportionality_or_disproportionality: ProportionalityOrDisproportionality | None = None,
        loyalty_or_betrayal: LoyaltyOrBetrayal | None = None,
        purity_or_degradation: PurityOrDegradation | None = None
    ):
        self.authority_or_subversion = authority_or_subversion
        self.care_or_harm = care_or_harm
        self.equality_or_inequality = equality_or_inequality
        self.proportionality_or_disproportionality = proportionality_or_disproportionality
        self.loyalty_or_betrayal = loyalty_or_betrayal
        self.purity_or_degradation = purity_or_degradation
