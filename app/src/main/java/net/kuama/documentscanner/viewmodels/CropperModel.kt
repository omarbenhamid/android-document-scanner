package net.kuama.documentscanner.viewmodels

import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import net.kuama.documentscanner.data.Corners
import net.kuama.documentscanner.data.CornersFactory
import net.kuama.documentscanner.support.Failure
import net.kuama.documentscanner.domain.FindPaperSheetContours
import net.kuama.documentscanner.domain.PerspectiveTransform
import net.kuama.documentscanner.domain.UriToBitmap
import net.kuama.documentscanner.utils.PointUtils
import org.opencv.core.Point
import org.opencv.core.Size

class CropperModel : ViewModel() {
    private val perspectiveTransform: PerspectiveTransform = PerspectiveTransform()
    private val findPaperSheetUseCase: FindPaperSheetContours = FindPaperSheetContours()
    private val uriToBitmap: UriToBitmap = UriToBitmap()

    val corners = MutableLiveData<Corners>()
    val originalBitmap = MutableLiveData<Bitmap>()
    val bitmapToCrop = MutableLiveData<Bitmap>()
    val errors = MutableLiveData<Throwable>()

    fun onViewCreated(uri: Uri, screenOrientationDeg: Int) {
        viewModelScope.launch {
            uriToBitmap(
                UriToBitmap.Params(
                    uri = uri,
                    screenOrientationDeg = screenOrientationDeg
                )
            ) { either ->
                either.fold(::handleFailure) { preview ->
                    originalBitmap.value = preview
                    analyze(preview)
                }
            }
        }
    }

    fun onCornersAccepted(bitmap: Bitmap) {
        corners.value?.let { acceptedCorners ->
            val acceptedAndOrderedCorners = listOf(
                acceptedCorners.topLeft,
                acceptedCorners.topRight,
                acceptedCorners.bottomRight,
                acceptedCorners.bottomLeft
            )

            viewModelScope.launch {
                perspectiveTransform(
                    PerspectiveTransform.Params(
                        bitmap = bitmap,
                        corners = CornersFactory.create(
                            acceptedAndOrderedCorners,
                            acceptedCorners.size
                        )
                    )
                ) { bitmap ->
                    bitmapToCrop.value = bitmap
                }
            }
        }
    }

    private fun analyze(bitmap: Bitmap) {
        viewModelScope.launch {
            findPaperSheetUseCase(
                FindPaperSheetContours.Params(bitmap)
            ) { foundCorners: Corners? ->
                corners.value = foundCorners ?: PointUtils.getSortedCorners(
                    listOf(Point(0.0,0.0), Point(0.0, bitmap.height.toDouble()),
                        Point(bitmap.width.toDouble(), bitmap.height.toDouble()),
                        Point(bitmap.width.toDouble(), 0.0)),
                    Size(bitmap.width.toDouble(), bitmap.height.toDouble()))
            }
        }
    }

    private fun handleFailure(failure: Failure) {
        errors.value = failure.origin
    }
}
