# kb/tax_engine.py
from logic import Symbol, And, Not, Implication, model_check

class TaxEngine:
    def __init__(self):
        # Define predicates
        self.M = Symbol('IsMarried')           # Married status
        self.J = Symbol('JointFiling')         # Joint filing status
        self.C = Symbol('HasChildren')         # Presence of children
        self.SI = Symbol('SpouseIncome')       # Spouse has income

        # Conclusion symbols
        self.BaseExemption                 = Symbol('BaseExemption')
        self.BaseExemptionWithChildren     = Symbol('BaseExemptionWithChildren')
        self.FullExemption                 = Symbol('FullExemption')
        self.FullExemptionWithoutChildren  = Symbol('FullExemptionWithoutChildren')
        self.FullExemptionWithChildren     = Symbol('FullExemptionWithChildren')

        # Build rule set reflecting Jordanian tax law
        self.rules = []
        # 1. Unmarried ⇒ BaseExemption
        self.rules.append(Implication(Not(self.M), self.BaseExemption))
        # 2. Married ∧ JointFiling ∧ ¬HasChildren ∧ ¬SpouseIncome ⇒ FullExemption
        self.rules.append(Implication(And(self.M, self.J, Not(self.C), Not(self.SI)),
                                      self.FullExemption))
        # 3. Married ∧ JointFiling ∧ ¬HasChildren ∧ SpouseIncome ⇒ FullExemptionWithoutChildren
        self.rules.append(Implication(And(self.M, self.J, Not(self.C), self.SI),
                                      self.FullExemptionWithoutChildren))
        # 4. Married ∧ JointFiling ∧ HasChildren ∧ SpouseIncome ⇒ FullExemptionWithChildren
        self.rules.append(Implication(And(self.M, self.J, self.C, self.SI),
                                      self.FullExemptionWithChildren))
        # 5. Married ∧ ¬JointFiling ∧ HasChildren ⇒ BaseExemptionWithChildren
        self.rules.append(Implication(And(self.M, Not(self.J), self.C),
                                      self.BaseExemptionWithChildren))
        # 6. Married ∧ ¬JointFiling ∧ ¬HasChildren ⇒ BaseExemption
        self.rules.append(Implication(And(self.M, Not(self.J), Not(self.C)),
                                      self.BaseExemption))

    def evaluate(self, facts: dict):
        """
        Determines exemption category based on input facts.

        facts keys:
          - 'married' (bool)
          - 'children' (int)
          - 'spouse_income' (bool)
          - 'joint_filing' (bool)

        Returns a string naming the applicable exemption.
        """
        # Assemble KB with all rules
        kb = And(*self.rules)

        # Assert input facts
        kb.add(self.M if facts.get('married') else Not(self.M))
        kb.add(self.C if facts.get('children', 0) > 0 else Not(self.C))
        kb.add(self.SI if facts.get('spouse_income') else Not(self.SI))
        kb.add(self.J if facts.get('joint_filing') else Not(self.J))

        # Check in priority order
        if model_check(kb, self.FullExemptionWithChildren):
            return self.FullExemptionWithChildren.name
        if model_check(kb, self.FullExemptionWithoutChildren):
            return self.FullExemptionWithoutChildren.name
        if model_check(kb, self.FullExemption):
            return self.FullExemption.name
        if model_check(kb, self.BaseExemptionWithChildren):
            return self.BaseExemptionWithChildren.name
        if model_check(kb, self.BaseExemption):
            return self.BaseExemption.name

        return 'Unknown'