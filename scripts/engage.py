#!/usr/bin/env python3
"""Engage the rl_policy controller: TRAJECTORY mode, then wait for the pilot to arm and go offboard.

With --arm it requests arming and offboard itself, for a simulator whose platform takes them.
"""

import argparse
import math
import sys
import time

from as2_msgs.msg import ControlMode, PlatformInfo
from as2_msgs.srv import SetControlMode
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.utilities import remove_ros_args
from std_srvs.srv import SetBool

CALL_TIMEOUT_S = 10.0
RETRY_PERIOD_S = 1.0
REMINDER_PERIOD_S = 10.0


class Engage(Node):
    """Service clients and the platform/info subscription of one drone."""

    def __init__(self, namespace: str, arm: bool):
        super().__init__('rl_policy_engage', namespace=namespace)
        self.info = None
        self.create_subscription(
            PlatformInfo, 'platform/info', self._on_info, QoSProfile(depth=10))
        self.set_control_mode = self.create_client(SetControlMode, 'controller/set_control_mode')
        self.set_arming_state = self.create_client(SetBool, 'set_arming_state') if arm else None
        self.set_offboard_mode = self.create_client(SetBool, 'set_offboard_mode') if arm else None

    def _on_info(self, msg: PlatformInfo) -> None:
        self.info = msg

    def spin_until(self, done, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if done():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return done()

    def wait(self, done, what: str, deadline: float) -> bool:
        """Spin until done() or the deadline, saying what it waits for now and then."""
        self.get_logger().info(f'Waiting for {what}')
        while rclpy.ok():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return done()
            if self.spin_until(done, min(REMINDER_PERIOD_S, remaining)):
                return True
            if deadline - time.monotonic() > 0.0:
                self.get_logger().info(f'Still waiting for {what}')
        return False

    def call(self, client, request, timeout: float):
        future = client.call_async(request)
        if not self.spin_until(future.done, timeout):
            return None
        return future.result()


def fail(node: Engage, message: str) -> int:
    node.get_logger().error(message)
    return 1


def flag(node: Engage, name: str) -> bool:
    return node.info is not None and bool(getattr(node.info, name))


def call_timeout(deadline: float) -> float:
    return max(min(CALL_TIMEOUT_S, deadline - time.monotonic()), 0.1)


def set_trajectory_mode(node: Engage, deadline: float) -> bool:
    request = SetControlMode.Request()
    request.control_mode.control_mode = ControlMode.TRAJECTORY
    request.control_mode.yaw_mode = ControlMode.YAW_ANGLE
    while rclpy.ok():
        response = node.call(node.set_control_mode, request, call_timeout(deadline))
        if response is not None and response.success:
            node.get_logger().info('Controller in TRAJECTORY mode')
            return True
        if time.monotonic() >= deadline:
            return False
        node.get_logger().warn('controller/set_control_mode failed, retrying')
        node.spin_until(lambda: False, RETRY_PERIOD_S)
    return False


def request_state(node: Engage, name: str, client, label: str, deadline: float) -> int:
    if flag(node, name):
        node.get_logger().info(f'Platform already {name}')
        return 0
    response = node.call(client, SetBool.Request(data=True), call_timeout(deadline))
    # A refused call is fine when the platform reports the state anyway.
    if not node.spin_until(lambda: flag(node, name), 2.0):
        detail = 'no response' if response is None else 'the platform refused'
        return fail(node, f'{label} failed: {detail}, platform/info still not {name}')
    node.get_logger().info(f'Platform {name}')
    return 0


def engage(node: Engage, timeout: float, arm: bool) -> int:
    ns = node.get_namespace()
    deadline = time.monotonic() + timeout
    clients = [(node.set_control_mode, 'controller')]
    if arm:
        clients += [(node.set_arming_state, 'platform'), (node.set_offboard_mode, 'platform')]
    for client, owner in clients:
        service = f'{ns.rstrip("/")}/{client.srv_name}'
        if not node.wait(lambda c=client: c.service_is_ready(), service, deadline):
            return fail(node, f'{service} is not available: is the {owner} running in {ns}?')
    if not node.wait(lambda: node.info is not None, f'{ns}/platform/info', deadline):
        return fail(node, f'no message on {ns}/platform/info')

    if not set_trajectory_mode(node, deadline):
        return fail(node, 'controller/set_control_mode(TRAJECTORY, YAW_ANGLE) kept failing: '
                          'check the controller log for the refusal')

    if arm:
        for name, client, label in (('armed', node.set_arming_state, 'arm'),
                                    ('offboard', node.set_offboard_mode, 'offboard')):
            code = request_state(node, name, client, label, deadline)
            if code != 0:
                return code
        return 0

    if not node.wait(lambda: flag(node, 'armed'), 'the pilot to arm', deadline):
        return fail(node, 'the platform was not armed in time')
    node.get_logger().info('Platform armed')
    if not node.wait(lambda: flag(node, 'armed') and flag(node, 'offboard'),
                     'the pilot to switch to offboard', deadline):
        return fail(node, 'the platform was not armed and offboard in time')
    node.get_logger().info('Platform offboard: the controller flies')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='drone0', help='drone namespace')
    parser.add_argument('--timeout', type=float, default=math.inf, metavar='S',
                        help='give up after S seconds (default: wait until done)')
    parser.add_argument('--arm', action='store_true',
                        help='request arming and offboard after the mode (simulator only)')
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = Engage(args.namespace, args.arm)
    try:
        code = engage(node, args.timeout, args.arm)
    except KeyboardInterrupt:
        code = 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
