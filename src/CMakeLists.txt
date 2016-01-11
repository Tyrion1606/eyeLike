SET(LIBRARIES ${LIBRARIES} ${OpenCV_LIBS})

ADD_EXECUTABLE( eyeLike main.cpp findEyeCenter.cpp findEyeCenter.h findEyeCorner.cpp findEyeCorner.h constants.h helpers.cpp helpers.h dialer.cpp dialer.h sound.cpp sound.h)
TARGET_LINK_LIBRARIES( eyeLike ${LIBRARIES} )
