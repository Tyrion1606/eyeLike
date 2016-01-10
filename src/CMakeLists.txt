#find_library(MGL mgl PATHS /usr/local/lib)
FIND_PACKAGE(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
	message(ERROR "OPENGL not found")
endif(NOT OPENGL_FOUND)
FIND_PACKAGE(GLUT REQUIRED)
if(NOT GLUT_FOUND)
	message(ERROR "GLUT not found")
endif(NOT GLUT_FOUND)
INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

SET(LIBRARIES ${LIBRARIES} ${OPENGL_LIBRARIES})
SET(LIBRARIES ${LIBRARIES} ${OpenCV_LIBS})

ADD_EXECUTABLE( eyeLike main.cpp findEyeCenter.cpp findEyeCenter.h findEyeCorner.cpp findEyeCorner.h constants.h helpers.cpp helpers.h dialer.cpp dialer.h)
TARGET_LINK_LIBRARIES( eyeLike ${LIBRARIES} pthread)
#target_link_libraries( eyeLike ${MGL})



