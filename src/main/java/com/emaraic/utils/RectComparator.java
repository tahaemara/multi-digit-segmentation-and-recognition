package com.emaraic.utils;

import java.util.Comparator;
import org.bytedeco.javacpp.opencv_core;

/**
 *
 * @author Taha Emara
 * Website: http://www.emaraic.com
 * Email  : taha@emaraic.com
 * Created on: Feb 14, 2018 
 */
 public class RectComparator implements Comparator<opencv_core.Rect> {

        @Override
        public int compare(opencv_core.Rect t1, opencv_core.Rect t2) {
            return Integer.valueOf(t1.x()).compareTo(t2.x());
        }
}
