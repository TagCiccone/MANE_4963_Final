cd /home/ros2_ws

mkdir -p lib
cd lib

if [ ! -d "sim_ros_framework" ]; then
    git clone https://github.com/xal-rpi/sim_ros_framework.git
fi
cd sim_ros_framework
git fetch
git switch hpa-s26
git pull -f

./luamod/build.bash --win --all


. /opt/ros/jazzy/setup.sh
./xal_bng_ws_build.bash -w /home/ros2_ws -r jazzy --clean



echo "source /home/ros2_ws/install/setup.bash" >> ~/.bashrc

cd /home/ros2_ws
