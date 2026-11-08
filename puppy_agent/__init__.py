"""PuppyPi agent: step executor, navigation and perception, independent of Isaac Sim.

Nothing in this package imports Isaac Sim, ROS 2 or torch, so it can be unit-tested anywhere.
The simulator (sim/) and the VLM server (vlm_server/) plug in through small interfaces.
"""
