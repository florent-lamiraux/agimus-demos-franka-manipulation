# Copyright 2023 CNRS
# Author: Florent Lamiraux
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from hpp.corbaserver import Client, loadServerPlugin, shrinkJointRange, \
    wrap_delete
from hpp.corbaserver.bin_picking import Client as BpClient
from hpp.corbaserver.manipulation import Client as ManipClient, ProblemSolver
from hpp.corbaserver.manipulation import Constraints, Robot, Rule
from hpp.corbaserver.problem_solver import _convertToCorbaAny as convertToAny
from agimus_demos import InStatePlanner
from create_graph import makeGraph
from agimus_demos.tools_hpp import concatenatePaths
from tools_hpp import displayGripper, displayHandle, generateTargetConfig, \
    shootPartInBox

## Write a handle in a string in srdf format
def writeHandleInSrdf(robot, handle, clearance, mask):
    joint, pose = robot.getHandlePositionInJoint(handle)
    link = robot.getLinkNames(joint)[0]
    # Guess prefix
    rank = handle.find("/")
    if rank == -1:
        prefix = ""
        handle_short = handle
        link_short = link
    else:
        prefix = handle[:rank]
        handle_short = handle[rank+1:]
        if link[:rank] != prefix:
            raise RuntimeError(
                f"Expected prefix {prefix}/ in link name, but got {link}")
        link_short = link[rank+1:]
    res = f'''
  <handle name="{handle_short}" clearance="{clearance}">
    <position xyz="{pose[0]} {pose[1]} {pose[2]}" wxyz="{pose[6]} {pose[3]} {pose[4]} {pose[5]}"/>
    <link name="{link_short}"/>
    <mask>{mask[0]} {mask[1]} {mask[2]} {mask[3]} {mask[4]} {mask[5]}</mask>
  </handle>
'''
    return res
## This class defines a bin-picking problem.
#
class BinPicking(object):
    objects = list()
    """
    List of object names. The part to grasp is the first one.
    """
    robotGrippers = list()
    """
    List of robot grippers to grasp the object
    """
    goalGrippers = list()
    """
    List of gripper that define goal poses of the object, ranked in order of
    priority. Using only one robot gripper, it may not be possible to release
    the object always in the same pose.
    """
    nh = 16
    """
    Number of handles to produce when discretizing
    """
    handles = list()
    """
    List of handles of the object to grasp
    """
    handlesToDiscretize = list()
    """
    List of handles to discretize
    """
    goalHandles = list()
    """
    List of handles that define the goal poses of the object. mapped one to
    one with goalGrippers.
    """
    graphConstraints = list()
    """
    List of constraints to be added to the constraint graph: quasi-static
    equilibrium, gripper open...
    """
    goalConfigs = dict()
    """
    List of precomputed goal configurations to release the object
    """
    q_goal = None
    """
    Configuration with which goal configurations are precomputed.
    This configuration defines the poses of objects other than the part.
    """
    def wd(self, o):
        return wrap_delete(o, self.ps.client.basic._tools)

    def __init__(self, ps):
        self.ps = ps
        self.robot = ps.robot
        self.bpc = BpClient()
        # Store pregrasp configurations for releasing the object
        # self.goalConfig["pregrasp"][gripper][handle] = config
        # where config is the configuration to reach when the object is
        # grasped by (gripper, handle)
        self.goalConfigs["pregrasp"] = dict()
        # Store grasp configurations for releasing the object
        self.goalConfigs["grasp"] = dict()
        # Store preplace configurations for releasing the object
        self.goalConfigs["preplace"] = dict()
        # store which pair (goalGripper, goalHandle) corresponds to the
        # goal configuration when the object is grasped by (gripper, handle)
        # self.goalGrasps[gripper][handle] = i, where
        # goalGripper = self.goalGrippers[i]
        # goalHandle = self.goalHandles[i]
        self.goalGrasps = dict()


    def _rules(self):
        """
        Build rules for the constraint graph factory
        """
        rules = list()
        for ig, gripper in enumerate(self.goalGrippers):
            for ih, handle in enumerate(self.goalHandles):
                if ig != ih:
                    rules  += [
                        Rule(grippers=[gripper], handles = [handle], link=False)
                        ]
        # object cannot be placed in two goal positions at the same time
        rules += [
            Rule(grippers=self.robotGrippers + self.goalGrippers,
                 handles = [".*", "part/center1", "part/center2"],
                 link=False),
            Rule(grippers=[".*"], handles=[".*"], link=True)
        ]
        return rules

    def _possibleGrasps(self):
        res = dict()
        # Each goal gripper can only grasp one goal handle
        if len(self.goalGrippers) != len(self.goalHandles):
            raise RuntimeError(
                f"number of goal grippers ({len(self.goalGrippers)})"
                + " should be the same as " +
                f"number of goal handles ({len(self.goalHandles)})"
            )
        for g,h in zip(self.goalGrippers, self.goalHandles):
            res[g] = [h]
        for g in self.robotGrippers:
            res[g] = self.handles
        return res

    def writeRules(self, f):
        """
        Write the rules in a stream
        """
        f.write("rules:\n")
        for r in self._rules():
            f.write(f"  - grippers: {r.grippers}\n" +
                    f"    handles: {r.handles}\n" +
                    f"    link: {r.link}\n")

    def writeTranstions(self, f):
        """
        Write the edges in a yaml file
        """
        f.write("transitions:\n")
        for e in self.graph.edges.keys():
            f.write(f'  - "{e}"\n')

    def writeStates(self, f):
        """
        Write the nodes of the constraint graph in a yaml file
        """
        f.write("states:\n")
        for s in self.graph.nodes.keys():
            f.write(f'  - "{s}"\n')

    def buildGraph(self):
        """
        Build the constraint graph
          - discretize handles,
        """
        # store list of not discretized handles
        self.initialHandles = self.handles[:]
        # Discretize handles
        for h in self.handlesToDiscretize:
            self.bpc.bin_picking.discretizeHandle(h, self.nh)
            self.handles += ['%s_%03d' % (h, i) for i in range(self.nh)]
        # build graph
        # Sort handles in alphabetic order to be sure that the order is the
        # same in agimus-sot.
        handles = self.handles + self.goalHandles
        handles.sort()
        self.factory, self.graph = makeGraph(
            self.ps, self.robot, self.robotGrippers + self.goalGrippers,
            self.objects, [handles, []], self._rules(), self._possibleGrasps())
        self.graph.addConstraints(graph=True,
            constraints = Constraints(numConstraints = self.graphConstraints))
        self.graph.initialize()
        self.transitionPlanner = self.wd(self.ps.client.manipulation.problem.\
                                      createTransitionPlanner())
        self.transitionPlanner.timeOut(3.)
        self.transitionPlanner.setPathProjector('Progressive', 0.2)
        self.transitionPlanner.addPathOptimizer("EnforceTransitionSemantic")
        self.transitionPlanner.addPathOptimizer("SimpleTimeParameterization")
        self.transitionPlanner.setParameter('SimpleTimeParameterization/order',
                                         convertToAny(2))
        self.transitionPlanner.setParameter(
            'SimpleTimeParameterization/maxAcceleration', convertToAny(1.))
        self.transitionPlanner.setParameter(
            'SimpleTimeParameterization/safety', convertToAny(.95))
    def buildEffectors(self, obstacles, q):
        """
        build and effector to test collision of grasps
          - name: name of the effector,
          - gripper: name of the gripper,
          - obstacles: list of obstacle name with which the effector with
                       be tested for collision,
          - q: any configuration where the various geometries of the gripper are
               in the right relative pose.
        """
        for gripper in self.robotGrippers:
            # find any edge that grasps the object
            edge = None
            for e in self.graph.edges.keys():
                if e.startswith(gripper + " > ") and e.endswith("12"):
                    edge = e
                    break
            if edge is None:
                raise RuntimeError(
                    f"Did not find any edge where {gripper} grasps anything")
            self.bpc.bin_picking.createEffector(gripper, gripper, q,
                                                self.graph.edges[edge])
        for o in obstacles:
            self.bpc.bin_picking.addObstacleToEffector(gripper, o, 0.)

    def generateGoalConfigs(self, q):
        """
        Precompute goal configurations for the part hold by the robot
          - q configuration that provides the pose of other objects
        """
        self.q_goal = q[:]
        # Separate indices of handles and grippers to distinguish goal
        # grippers/handles from regular ones
        robotGrippers = list()
        goalGrippers = list()
        regularHandles = list()
        goalHandles = list()
        for i, h in enumerate(self.factory.handles):
            if h in self.handles:
                regularHandles.append(i)
            elif h in self.goalHandles:
                goalHandles.append(i)
            else:
                raise RuntimeError(
                    f"'{h}' is neither a regular handle nor a goal handle")
        for i, g in enumerate(self.factory.grippers):
            if g in self.robotGrippers:
                robotGrippers.append(i)
            elif g in self.goalGrippers:
                goalGrippers.append(i)
            else:
                raise RuntimeError(
                    f"'{g}' is neither a regular gripper nor a goal gripper")
        for irg in robotGrippers:
            robotGripper = self.factory.grippers[irg]
            self.goalConfigs["pregrasp"][robotGripper] = dict()
            self.goalConfigs["grasp"][robotGripper] = dict()
            self.goalConfigs["preplace"][robotGripper] = dict()
            self.goalGrasps[robotGripper] = dict()

            for ih in regularHandles:
                handle = self.factory.handles[ih]
                for igg, (goalGripper, goalHandle) in enumerate(
                        zip(self.goalGrippers, self.goalHandles)):
                    edge = f"{goalGripper} > {goalHandle} | {irg}-{ih}_01"
                    if not edge in self.graph.edges.keys(): breakpoint()
                    res, q1 = generateTargetConfig(self.robot, self.graph,
                                                   edge, q)
                    if not res: continue
                    edge = f"{goalGripper} > {goalHandle} | {irg}-{ih}_12"
                    if not edge in self.graph.edges.keys(): breakpoint()
                    res, q2 = generateTargetConfig(self.robot, self.graph,
                                                   edge, q1)
                    if not res: continue
                    ggIndex = self.factory.grippers.index(goalGripper)
                    ghIndex = self.factory.handles.index(goalHandle)
                    edge = f"{robotGripper} < {handle} | {irg}-{ih}:" + \
                        f"{ggIndex}-{ghIndex}_21"
                    if not edge in self.graph.edges.keys(): breakpoint()
                    res, q3 = generateTargetConfig(self.robot, self.graph,
                                                   edge, q2)
                    self.goalConfigs["pregrasp"][robotGripper][handle] = q1
                    self.goalConfigs["grasp"][robotGripper][handle] = q2
                    self.goalConfigs["preplace"][robotGripper][handle] = q3
                    self.goalGrasps[robotGripper][handle] = igg
                    if res: break

    def checkObjectPoses(self, q):
        """
        Check that
          - the initial configuration is in state "free",
          - each object is in the same pose as in the configuration
            that was used to precompute the goal configurations. Index 0
            corresponds to the part and thus is skipped.
        """
        if self.graph.getNode(q) != "free":
            raise RuntimeError(
                f"Initial configuration {q} not in state 'free'.")
        if not self.q_goal:
            raise RuntimeError(
                "You need to call method generateGoalConfigs first.")
        for o in self.objects[1:]:
            r = self.robot.rankInConfiguration[f"{o}/root_joint"]
            if q[r:r+7] != self.q_goal[r:r+7]:
                raise RuntimeError(f"Object {o} is in pose {q[r:r+7]} but " +
                    "was in pose {self.q_goal[r:r+7]} when pre-computing" +
                    " goal configurations.")

    def computeFreeGrasps(self, q):
        """
        For each gripper, compute list of object handles that are not in
        collision in the given configuration.

        Result is a dictionary of keys the robot grippers and with values
        lists of handles.
        """
        self._freeGrasps = dict()
        res = False
        for gripper in self.robotGrippers:
            freeGrasps = list()
            self._freeGrasps[gripper] = list()
            for handle in self.handles:
                col, msg, gripperAxis = self.bpc.bin_picking.collisionTest(
                    gripper, handle, q)
                if not col:
                    # Store pairs (handle, score)
                    freeGrasps.append((handle, -gripperAxis[2]))
                    res = True
            # Sort handles by increasing z coordinate of gripper axis
            l = sorted(freeGrasps, key = lambda x:x[1], reverse = True)
            if len(l) > 0:
                self._freeGrasps[gripper] = list(zip(*l))[0]
        return res

    def selectGrasp(self, q):
        """
        Select a grasp among all possible collision-free grasps
          - q initial configuration of the system,
          - return gripper, handle, and 6 waypoint configurations corresponding
              to - pregrasp, grasp, preplace configurations to grasp the object,
                 - pregrasp, grasp, preplace configurations to release the
                   object.
        """
        for gripper in self.robotGrippers:
            for handle in self._freeGrasps[gripper]:
                # check that goal configuration exists for this grasp
                q4 = self.goalConfigs["pregrasp"][gripper].get(handle)
                if q4 is None: continue
                q5 = self.goalConfigs["grasp"][gripper].get(handle)
                if q5 is None: continue
                q6 = self.goalConfigs["preplace"][gripper].get(handle)
                if q6 is None: continue
                # generate pregrasp, grasp and preplace configuration to
                # grasp the object
                edge = f"{gripper} > {handle} | f_01"
                res, q1 = generateTargetConfig(self.robot, self.graph, edge, q)
                if not res: continue
                edge = f"{gripper} > {handle} | f_12"
                res, q2 = generateTargetConfig(self.robot, self.graph, edge, q1)
                if not res: continue
                edge = f"{gripper} > {handle} | f_23"
                res, q3 = generateTargetConfig(self.robot, self.graph, edge, q2)
                if not res: continue
                if res:
                    return (gripper, handle, q1, q2, q3, q4, q5, q6)
        return 8*(None,)


    def solve(self, q):
        """
        Compute a trajectory to grasp and release the object
          - q initial configuration of the robot and objects
        """
        self.checkObjectPoses(q)
        res = self.computeFreeGrasps(q)
        if not res:
            msg = "End effector in collision for all possible grasps."
            return False, msg
        gripper, handle, q1, q2, q3, q4, q5, q6 = self.selectGrasp(q)
        if gripper is None:
            msg = "Failed to generate collision-free grasp with valid goal " +\
                "configurations"
            return False, msg
        # Plan paths between waypoint configurations
        edge = "Loop | f"
        self.transitionPlanner.setEdge(self.graph.edges[edge])
        try:
            p1 = self.transitionPlanner.planPath(q, [q1,], True)
        except Exception as exc:
            raise RuntimeError(f"Failed to connect {q} and {q1}: {exc}")
        edge = f"{gripper} > {handle} | f_12"
        self.transitionPlanner.setEdge(self.graph.edges[edge])
        p2, res, msg = self.transitionPlanner.directPath(q1, q2, False)
        assert(res)
        p2 = self.transitionPlanner.timeParameterization(p2.asVector())
        edge = f"{gripper} > {handle} | f_23"
        self.transitionPlanner.setEdge(self.graph.edges[edge])
        p3, res, msg = self.transitionPlanner.directPath(q2, q3, False)
        assert(res)
        p3 = self.transitionPlanner.timeParameterization(p3.asVector())
        ig = self.factory.grippers.index(gripper)
        ih = self.factory.handles.index(handle)
        edge = f"Loop | {ig}-{ih}"
        self.transitionPlanner.setEdge(self.graph.edges[edge])
        try:
            p4 = self.transitionPlanner.planPath(q3, [q4,], True)
        except Exception as exc:
            raise RuntimeError(f"Failed to connect {q3} and {q4}: {exc}")
        igg = self.goalGrasps[gripper][handle]
        ng = len(self.robotGrippers); nh = len(self.handles)
        goalGripper = self.goalGrippers[igg]
        goalHandle = self.goalHandles[igg]
        edge = f"{goalGripper} > {goalHandle} | {ig}-{ih}_12"
        self.transitionPlanner.setEdge(self.graph.edges[edge])
        p5, res, msg = self.transitionPlanner.directPath(q4, q5, False)
        assert(res)
        p5 = self.transitionPlanner.timeParameterization(p5.asVector())
        ggIndex = self.factory.grippers.index(goalGripper)
        ghIndex = self.factory.handles.index(goalHandle)
        edge = f"{gripper} < {handle} | {ig}-{ih}:{ggIndex}-{ghIndex}_21"
        self.transitionPlanner.setEdge(self.graph.edges[edge])
        p6, res, msg = self.transitionPlanner.directPath(q5, q6, False)
        assert(res)
        p6 = self.transitionPlanner.timeParameterization(p6.asVector())
        edge = "Loop | f"
        self.transitionPlanner.setEdge(self.graph.edges[edge])
        # Return to initial configuration
        q7 = q[:]
        r = self.robot.rankInConfiguration[f"{self.objects[0]}/root_joint"]
        q7[r:r+7] = q6[r:r+7]
        try:
            p7 = self.transitionPlanner.planPath(q6, [q7,], True)
        except Exception as exc:
            raise RuntimeError(f"Failed to connect {q6} and {q7}: {exc}")
        return True, concatenatePaths([p1,p2,p3,p4,p5,p6,p7])

if __name__ == "__main__":
    from hpp.rostools import process_xacro, retrieve_resource
    Robot.urdfString = process_xacro\
        ("package://agimus_demos/franka/manipulation/urdf/demo.urdf.xacro")
    Robot.srdfString = ""

    problems = ["default", "learning"]
    defaultContext = "corbaserver"
    loadServerPlugin (defaultContext, "manipulation-corba.so")
    cl = ManipClient()
    for p in problems:
        res = cl.problem.selectProblem(p)
        if res: print(f'created problem "{p}"')
        else: print(f'did not create problem "{p}"')
        cl.problem.resetProblem()

    cl.problem.selectProblem("default")
    robot = Robot("robot", "pandas", rootJointType="anchor")
    shrinkJointRange(robot, [f'pandas/panda2_joint{i}' for i in range(1,8)],
                     0.95)
    ps = ProblemSolver(robot)
