package net.kuama.documentscanner.domain

import android.graphics.Bitmap
import android.util.Log
import net.kuama.documentscanner.data.Corners
import net.kuama.documentscanner.data.CornersFactory
import net.kuama.documentscanner.extensions.shape
import net.kuama.documentscanner.support.*
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

class FindPaperSheetContours : InfallibleUseCase<Corners?, FindPaperSheetContours.Params>() {

    class Params(val bitmap: Bitmap)

    override suspend fun run(params: Params): Corners? {
        val original = Mat()
        val modified = Mat()

        Utils.bitmapToMat(params.bitmap, original)

        // Convert image from RGBA to GrayScale
        Imgproc.cvtColor(original, modified, Imgproc.COLOR_RGBA2GRAY)


        // Strong Gaussian Filter
        Imgproc.GaussianBlur(modified, modified, Size(51.0, 51.0), 0.0)

        // Thersolding
        //Imgproc.threshold(modified, modified, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU )


        // Canny Edge Detection
        Imgproc.Canny(modified, modified, 100.0, 200.0, 5, false)

        // Closing: Dilation followed by Erosion
        Imgproc.dilate(
            modified, modified, Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, Size(8.0, 8.0)
            )
        )
        Imgproc.erode(
            modified, modified, Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, Size(3.0, 3.0)
            )
        )

        var contours: MutableList<MatOfPoint> = ArrayList()
        val hierarchy = Mat()
        Imgproc.findContours(
            modified, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE
        )

        hierarchy.release()

        var minarea=original.width()*original.height()/10;

        contours = contours.map {
            var out = MatOfPoint2f()
            var c = MatOfPoint2f()
            it.convertTo(c, CvType.CV_32F)
            Imgproc.approxPolyDP(c, out, Imgproc.arcLength(c, true)*0.02,true)
            var ret=MatOfPoint()
            out.convertTo(ret, CvType.CV_32S)
            c.release()
            out.release()
            ret}
            .filter { it.shape.size == 4
                    && Imgproc.isContourConvex(it)
                    && Imgproc.contourArea(it) > minarea} //&& Imgproc.somethingSure does not exist(it.shape)
            .toTypedArray()
            .toMutableList()

        contours.sortWith { lhs, rhs ->
            Imgproc.contourArea(rhs).compareTo(Imgproc.contourArea(lhs))
        }

        val result: Corners? = contours.firstOrNull()?.let {
            CornersFactory.create(it.shape, original.size())
        }

        return result
    }
}
