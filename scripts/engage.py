#!/usr/bin/env python3
"""Engage the rl_policy controller: TRAJECTORY mode, then armed and offboard."""

import argparse
import sys
import time

from as2_msgs.msg import ControlMode, PlatformInfo
from as2_msgs.srv import SetControlMode
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.utilities import remove_ros_args
from std_srvs.srv import SetBool


class Engage(Node):
    """Service clients and the platform/info subscription of one drone."""

    def __init__(self, namespace: str):
        super().__init__('rl_policy_engage', namespace=namespace)
        self.info = None
        self.create_subscription(
            PlatformInfo, 'platform/info', self._on_info, QoSProfile(depth=10))
        self.set_control_mode = self.create_client(SetControlMode, 'controller/set_control_mode')
        self.set_arming_state = self.create_client(SetBool, 'set_arming_state')
        self.set_offboard_mode = self.create_client(SetBool, 'set_offboard_mode')

    def _on_info(self, msg: PlatformInfo) -> None:
        self.info = msg

    def spin_until(self, done, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if done():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return done()

    def call(self, client, request, timeout: float):
        future = client.call_async(request)
        if not self.spin_until(future.done, timeout):
            return None
        return future.result()


def fail(node: Engage, message: str) -> int:
    node.get_logger().error(message)
    return 1


def engage(node: Engage, timeout: float) -> int:
    ns = node.get_namespace()
    for client, owner in ((node.set_control_mode, 'controller'),
                          (node.set_arming_state, 'platform'),
                          (node.set_offboard_mode, 'platform')):
        node.get_logger().info(f'Waiting for {client.srv_name}')
        if not client.wait_for_service(timeout_sec=timeout):
            return fail(node, f'{client.srv_name} is not available after {timeout:.0f} s: '
                              f'is the {owner} running in {ns}?')
    node.get_logger().info(f'Waiting for {ns}/platform/info')
    if not node.spin_until(lambda: node.info is not None, timeout):
        return fail(node, f'no message on {ns}/platform/info after {timeout:.0f} s')

    request = SetControlMode.Request()
    request.control_mode.control_mode = ControlMode.TRAJECTORY
    request.control_mode.yaw_mode = ControlMode.YAW_ANGLE
    deadline = time.monotonic() + timeout
    while True:
        response = node.call(node.set_control_mode, request, max(deadline - time.monotonic(), 0.1))
        if response is not None and response.success:
            node.get_logger().info('Controller in TRAJECTORY mode')
            break
        if time.monotonic() >= deadline:
            return fail(node, 'controller/set_control_mode(TRAJECTORY, YAW_ANGLE) kept failing: '
                              'check the controller log for the refusal')
        node.get_logger().warn('controller/set_control_mode failed, retrying')
        node.spin_until(lambda: False, 1.0)

    for flag, client, label in (('armed', node.set_arming_state, 'arm'),
                                ('offboard', node.set_offboard_mode, 'offboard')):
        if getattr(node.info, flag):
            node.get_logger().info(f'Platform already {flag}')
            continue
        response = node.call(client, SetBool.Request(data=True), timeout)
        # A refused call is fine when the platform reports the state anyway.
        if not node.spin_until(lambda: getattr(node.info, flag), 2.0):
            detail = 'no response' if response is None else 'the platform refused'
            return fail(node, f'{label} failed: {detail}, platform/info still not {flag}')
        node.get_logger().info(f'Platform {flag}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='drone0', help='drone namespace')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='seconds to wait for each service, message and call')
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = Engage(args.namespace)
    try:
        code = engage(node, args.timeout)
    except KeyboardInterrupt:
        code = 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
