import re

sim_path = "backend/engine/simulator.py"
with open(sim_path, "r") as f:
    content = f.read()

# We want to make sure RNG draws are independent of the length of action_dict
# The subagent wrote: "Processes features in sorted order to guarantee deterministic CRN stream consumption."
# Let's inspect the ActionExecutionModel class specifically.
import ast
class V(ast.NodeVisitor):
    def visit_ClassDef(self, node):
        if node.name == 'ActionExecutionModel':
            print("ActionExecutionModel source:")
            # get source
            print(ast.get_source_segment(content, node))
        self.generic_visit(node)

V().visit(ast.parse(content))
