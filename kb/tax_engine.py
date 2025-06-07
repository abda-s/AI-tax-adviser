# kb/tax_engine.py
from logic import Symbol, And, Not, Implication, model_check

class TaxEngine:
    def __init__(self):
        # Define predicates
        self.M = Symbol('IsMarried')           # Married status
        self.J = Symbol('JointFiling')         # Joint filing status
        self.C = Symbol('HasChildren')         # Presence of children
        self.WI = Symbol('WifeIncome')       # Wife has income

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
        # 2. Married ∧ JointFiling ∧ ¬HasChildren ∧ ¬WifeIncome ⇒ FullExemption
        self.rules.append(Implication(And(self.M, self.J, Not(self.C), Not(self.WI)),
                                      self.FullExemption))
        # 3. Married ∧ JointFiling ∧ ¬HasChildren ∧ WifeIncome ⇒ FullExemptionWithoutChildren
        self.rules.append(Implication(And(self.M, self.J, Not(self.C), self.WI),
                                      self.FullExemptionWithoutChildren))
        # 4. Married ∧ JointFiling ∧ HasChildren ∧ WifeIncome ⇒ FullExemptionWithChildren
        self.rules.append(Implication(And(self.M, self.J, self.C, self.WI),
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
          - 'wife_income' (bool)
          - 'joint_filing' (bool)

        Returns a string naming the applicable exemption.
        """
        # Assemble KB with all rules
        kb = And(*self.rules)

        # Assert input facts
        print("\n--- Tax Engine Evaluation ---")
        print("Raw facts from answers:", facts)

        is_married = facts.get('married')
        has_children = facts.get('children', 0) > 0
        has_wife_income = facts.get('wife_income')
        is_joint_filing = facts.get('joint_filing')

        if is_married: kb.add(self.M)
        else: kb.add(Not(self.M))
        print(f"  IsMarried: {is_married}")

        if has_children: kb.add(self.C)
        else: kb.add(Not(self.C))
        print(f"  HasChildren: {has_children}")

        if has_wife_income: kb.add(self.WI)
        else: kb.add(Not(self.WI))
        print(f"  WifeIncome: {has_wife_income}")

        if is_joint_filing: kb.add(self.J)
        else: kb.add(Not(self.J))
        print(f"  JointFiling: {is_joint_filing}")

        print("Knowledge Base (KB) after asserting facts:", kb)

        # Check in priority order
        print("\nChecking exemption rules:")
        if model_check(kb, self.FullExemptionWithChildren):
            print("  -> FullExemptionWithChildren is TRUE")
            return self.FullExemptionWithChildren.name
        print("  -> FullExemptionWithChildren is FALSE")

        if model_check(kb, self.FullExemptionWithoutChildren):
            print("  -> FullExemptionWithoutChildren is TRUE")
            return self.FullExemptionWithoutChildren.name
        print("  -> FullExemptionWithoutChildren is FALSE")

        if model_check(kb, self.FullExemption):
            print("  -> FullExemption is TRUE")
            return self.FullExemption.name
        print("  -> FullExemption is FALSE")

        if model_check(kb, self.BaseExemptionWithChildren):
            print("  -> BaseExemptionWithChildren is TRUE")
            return self.BaseExemptionWithChildren.name
        print("  -> BaseExemptionWithChildren is FALSE")

        if model_check(kb, self.BaseExemption):
            print("  -> BaseExemption is TRUE")
            return self.BaseExemption.name
        print("  -> BaseExemption is FALSE")

        print("No matching exemption found.")
        return 'Unknown'