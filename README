# Robotic Guard Dog

**Authors:** Gabriel Gladstone, Alexander Sosnkowski, Ryland Buck, Kyle McDonald  
**Course:** Autonomous Mobile Robots  

## Overview
The **Robotic Guard Dog** is an autonomous patrol system designed to secure bounded environments using advanced motion planning and real-time area coverage. The project reduces dependency on fixed surveillance systems, making it ideal for temporary or remote environments.

## Key Features
- **Efficient Mapping**: Utilizes SLAM to map environments by remote controller.  
- **Path Optimization**: Implements a skeleton generation algorithm (via Voronoi diagrams).  
- **Intruder Detection**: Incorporates IR sensors for real-time obstacle/intruder alerts.  
- **Robust Algorithms**: Combines graph theory, nearest-neighbor heuristics, and A* for efficient patrol paths.

## System Process
1. **Mapping Phase**: Manual control to generate a SLAM-based environment map.  
2. **Path Planning**: Conversion of grid maps into topological skeletons using EVG-THIN, followed by vertex graph creation for optimal routing.  
3. **Patrol Phase**: Continuous path execution with real-time obstacle monitoring.  
4. **Pursue State**: Alerts triggered upon detecting intrusions.

## Results
- Successfully tested in complex and varied environments.
- Demonstrated complete area coverage bounded by grid map size and real-time obstacle detection capabilities.

## Future Enhancements
- **Autonomous Exploration**: Enable initial mapping without user input.  
- **Advanced Intrusion Detection**: Integrate computer vision or machine learning for enhanced recognition.  
- **Dynamic Optimization**: Refine path planning for faster and more optimal coverage.  

## References
1. SLAM Toolbox for mapping.  
2. Portugal & Rocha's method for graph extraction from grid maps.  
3. EVG-THIN tool for skeleton generation.  
