/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "CudaBayerDemosaicConsumer.h"

namespace ArgusSamples
{

CudaBayerDemosaicStream::CudaBayerDemosaicStream(EGLDisplay display,
                                                 EGLStreamKHR stream,
                                                 Argus::Size2D<uint32_t> size)
    : m_eglDisplay(display)
    , m_bayerInputStream(stream)
    , m_bayerSize(size)
{
    // ARGB output size is half of the Bayer size after demosaicing.
    m_outputSize.width() = m_bayerSize.width() / 2;
    m_outputSize.height() = m_bayerSize.height() / 2;

    // Set the acquire timeout to infinite.
    eglStreamAttribKHR(m_eglDisplay, m_bayerInputStream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, -1);
}

CudaBayerDemosaicConsumer::CudaBayerDemosaicConsumer(EGLDisplay display,
                                                     std::vector<EGLStreamKHR> streams,
                                                     std::vector<Argus::Size2D<uint32_t>> sizes)
    : m_eglDisplay(display)
{
    for (unsigned int i = 0; i < streams.size(); i++) {
        CudaBayerDemosaicStream* stream = new
            CudaBayerDemosaicStream(display, streams[i], sizes[i]);
        m_streams.push_back(stream);
    }
}

CudaBayerDemosaicConsumer::~CudaBayerDemosaicConsumer()
{
    for (auto stream : m_streams) {
        delete stream;
    }
}

CudaBayerDemosaicStream::~CudaBayerDemosaicStream()
{
    shutdown();
}

bool CudaBayerDemosaicConsumer::threadInitialize()
{
    PROPAGATE_ERROR(initCUDA(&m_cudaContext));

    for (auto stream : m_streams) {
        PROPAGATE_ERROR(stream->initBeforePreview());
    }

    PROPAGATE_ERROR(initPreview());

    for (auto stream : m_streams) {
        PROPAGATE_ERROR(stream->initAfterPreview());
    }

    return true;
}

bool CudaBayerDemosaicStream::initBeforePreview()
{
    // Connect this CUDA consumer to the RAW16 Bayer stream being produced by Argus.
    CUresult cuResult = cuEGLStreamConsumerConnect(
            &m_cudaBayerStreamConnection, m_bayerInputStream);
    if (cuResult != CUDA_SUCCESS)
    {
        ORIGINATE_ERROR("Unable to connect CUDA to EGLStream as a consumer (CUresult %s)",
            getCudaErrorString(cuResult));
    }

    // Create the EGLStream for the demosaiced RGBA output to the OpenGL PreviewConsumer.
    if (!m_rgbaOutputStream.create(m_eglDisplay))
    {
        ORIGINATE_ERROR("Failed to create EGLStream for RGBA output");
    }

    return true;
}

bool CudaBayerDemosaicConsumer::initPreview()
{
    for (auto stream : m_streams) {
        m_rgbaOutputStreams.emplace_back(stream->getOutputStream());
    }

    // Connect the OpenGL PreviewConsumer to the RGBA stream.
    m_previewConsumerThread = new PreviewConsumerThread(
        m_eglDisplay, m_rgbaOutputStreams,
        PreviewConsumerThread::RenderLayout::LAYOUT_TILED);
    if (!m_previewConsumerThread)
    {
        ORIGINATE_ERROR("Failed to allocate preview consumer thread");
    }
    PROPAGATE_ERROR(m_previewConsumerThread->initialize());
    PROPAGATE_ERROR(m_previewConsumerThread->waitRunning());

    return true;
}

bool CudaBayerDemosaicStream::initAfterPreview()
{
    // Connect CUDA to the RGBA stream to procude the demosaiced output.
    CUresult cuResult = cuEGLStreamProducerConnect(
            &m_cudaRGBAStreamConnection, m_rgbaOutputStream.get(),
            m_outputSize.width(), m_outputSize.height());
    if (cuResult != CUDA_SUCCESS)
    {
        ORIGINATE_ERROR("Unable to connect CUDA to EGLStream as a producer (CUresult %s)",
            getCudaErrorString(cuResult));
    }

    // Allocate two RGBA buffers for double-buffering the EGLStream between CUDA and OpenGL.
    for (unsigned int i = 0; i < RGBA_BUFFER_COUNT; i++)
    {
        size_t bufferSize = m_outputSize.width() * m_outputSize.height() * sizeof(uint32_t);
        cuResult = cuMemAlloc(&m_rgbaBuffers[i], bufferSize);
        if (cuResult != CUDA_SUCCESS)
        {
            ORIGINATE_ERROR("Failed to allocate CUDA buffer");
        }
    }

    return true;
}

bool CudaBayerDemosaicConsumer::threadExecute()
{
    for (auto stream : m_streams) {
        PROPAGATE_ERROR(stream->initialize());
        PROPAGATE_ERROR(stream->waitRunning());
    }

    return true;
}

bool CudaBayerDemosaicStream::threadExecute()
{
    // Wait for the Argus producer to connect to the stream.
    while (true)
    {
        EGLint state = EGL_STREAM_STATE_CONNECTING_KHR;
        if (!eglQueryStreamKHR(m_eglDisplay, m_bayerInputStream, EGL_STREAM_STATE_KHR, &state))
        {
            ORIGINATE_ERROR("Failed to query stream state (possible producer failure).");
        }
        if (state == EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
        {
            break;
        }
        Window::getInstance().pollEvents();
    }

    // Render until there are no more frames (the producer has disconnected).
    uint32_t frame = 0;
    bool done = false;
    while (!done && !m_doShutdown)
    {
        bool newFrameAvailable = false;
        EGLint state;

        if (!eglQueryStreamKHR(m_eglDisplay, m_bayerInputStream, EGL_STREAM_STATE_KHR, &state) ||
            state == EGL_STREAM_STATE_DISCONNECTED_KHR)
        {
            REPORT_ERROR("Stream disconnected\n");
            done = true;
        }
        else if (state == EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
        {
            newFrameAvailable = true;
#if 0
            if (!eglStreamConsumerAcquireKHR(m_eglDisplay, m_bayerInputStream))
            {
                REPORT_ERROR("Failed to acquire stream\n");
                done = true;
            }
#endif
        }

        if (done || !newFrameAvailable)
        {
            continue;
        }

        // Acquire the new Bayer frame from the Argus EGLStream and get the CUDA resource.
        CUgraphicsResource bayerResource = 0;
        CUresult cuResult = cuEGLStreamConsumerAcquireFrame(
                &m_cudaBayerStreamConnection, &bayerResource, NULL, -1);
        if (cuResult != CUDA_SUCCESS)
        {
            ORIGINATE_ERROR("Unable to acquire an image frame from the EGLStream (CUresult %s)",
                getCudaErrorString(cuResult));
        }

        // Get the Bayer EGL frame information from the CUDA resource.
        CUeglFrame bayerEglFrame;
        memset(&bayerEglFrame, 0, sizeof(bayerEglFrame));
        cuResult = cuGraphicsResourceGetMappedEglFrame(&bayerEglFrame, bayerResource, 0, 0);
        if (cuResult != CUDA_SUCCESS)
        {
            ORIGINATE_ERROR("Unable to get the CUDA EGL frame (CUresult %s)",
                getCudaErrorString(cuResult));
        }

        // On the first frame, print the information contained in the CUDA EGL frame structure.
        if (frame == 0)
        {
            printf("CUDA CONSUMER:    Input frame format:\n");
            PROPAGATE_ERROR(printCUDAEGLFrame(bayerEglFrame));
        }

        // Sanity check for one of the required input color formats.
        if ((bayerEglFrame.cuFormat != CU_AD_FORMAT_SIGNED_INT16) ||
            ((bayerEglFrame.eglColorFormat != CU_EGL_COLOR_FORMAT_BAYER_RGGB) &&
             (bayerEglFrame.eglColorFormat != CU_EGL_COLOR_FORMAT_BAYER_BGGR) &&
             (bayerEglFrame.eglColorFormat != CU_EGL_COLOR_FORMAT_BAYER_GRBG) &&
             (bayerEglFrame.eglColorFormat != CU_EGL_COLOR_FORMAT_BAYER_GBRG)))
        {
            ORIGINATE_ERROR("Only 16bit signed Bayer color formats are supported");
        }

        // Prepare the next output frame for the RGBA stream. This will either be
        // an unused buffer or one that has been released by the OpenGL consumer.
        CUeglFrame rgbaEglFrame;
        if (frame < RGBA_BUFFER_COUNT)
        {
            // Populate a new CUeglFrame from one of the available buffers.
            rgbaEglFrame.frame.pPitch[0] = (void*)m_rgbaBuffers[frame];
            rgbaEglFrame.width = m_outputSize.width();
            rgbaEglFrame.height = m_outputSize.height();
            rgbaEglFrame.depth = 1;
            rgbaEglFrame.pitch = m_outputSize.width() * sizeof(uint32_t);
            rgbaEglFrame.frameType = CU_EGL_FRAME_TYPE_PITCH;
            rgbaEglFrame.planeCount = 1;
            rgbaEglFrame.numChannels = 4;
            rgbaEglFrame.eglColorFormat = CU_EGL_COLOR_FORMAT_ARGB;
            rgbaEglFrame.cuFormat = CU_AD_FORMAT_UNSIGNED_INT8;
            if (frame == 0)
            {
                printf("CUDA PRODUCER:    Output frame format:\n");
                PROPAGATE_ERROR(printCUDAEGLFrame(rgbaEglFrame));
            }
        }
        else
        {
            // Reuse a returned frame.
            cuResult = cuEGLStreamProducerReturnFrame(
                    &m_cudaRGBAStreamConnection, &rgbaEglFrame, NULL);
            if (cuResult != CUDA_SUCCESS)
            {
                ORIGINATE_ERROR("Unable to return frame to the CUDA producer(CUresult %s).",
                    getCudaErrorString(cuResult));
            }
        }

        printf("CUDA CONSUMER:    Acquired Bayer frame %d\n", frame + 1);

        // Run the CUDA kernel to demosaic the Bayer input into the RGBA output.
        cudaBayerDemosaic((CUdeviceptr)(bayerEglFrame.frame.pPitch[0]),
                          bayerEglFrame.width,
                          bayerEglFrame.height,
                          bayerEglFrame.pitch,
                          bayerEglFrame.eglColorFormat,
                          (CUdeviceptr)(rgbaEglFrame.frame.pPitch[0]));

        // Return the Bayer frame to the Argus stream.
        cuResult = cuEGLStreamConsumerReleaseFrame(
                &m_cudaBayerStreamConnection, bayerResource, NULL);
        if (cuResult != CUDA_SUCCESS)
        {
            ORIGINATE_ERROR("Unable to release frame to EGLStream (CUresult %s).",
                getCudaErrorString(cuResult));
        }

        // Present the RGBA output frame to the OpenGL PreviewConsumer stream.
        printf("CUDA PRODUCER:    Presented frame %d to RGBA EGLStream\n", frame + 1);
        cuResult = cuEGLStreamProducerPresentFrame(
                &m_cudaRGBAStreamConnection, rgbaEglFrame, NULL);
        if (cuResult != CUDA_SUCCESS)
        {
            ORIGINATE_ERROR("Failed to present frame to EGLStream (CUresult %s).",
                getCudaErrorString(cuResult));
        }

        frame++;
    }

    requestShutdown();

    printf("CUDA CONSUMER:    No more frames. Cleaning up\n");
    printf("CUDA CONSUMER:    Done\n");

    return true;
}

bool CudaBayerDemosaicStream::shutdownBeforePreview()
{
    // Disconnect from the Argus RAW16 stream
    CUresult cuResult = cuEGLStreamConsumerDisconnect(&m_cudaBayerStreamConnection);
    if (cuResult != CUDA_SUCCESS)
    {
        ORIGINATE_ERROR("Unable to disconnect CUDA from Bayer EGLStream (CUresult %s)",
            getCudaErrorString(cuResult));
    }

    // Disconnect the producer from the RGBA stream
    cuResult = cuEGLStreamProducerDisconnect(&m_cudaRGBAStreamConnection);
    if (cuResult != CUDA_SUCCESS)
    {
        ORIGINATE_ERROR("Unable to disconnect CUDA from RGBA EGLStream (CUresult %s)",
            getCudaErrorString(cuResult));
    }

    return true;
}

bool CudaBayerDemosaicConsumer::shutdownPreview()
{
    // Destroy the OpenGL consumer thread.
    m_previewConsumerThread->shutdown();
    delete m_previewConsumerThread;
    m_previewConsumerThread = NULL;

    return true;
}

bool CudaBayerDemosaicStream::shutdownAfterPreview()
{
    // Free the RGBA stream buffers.
    for (unsigned int i = 0; i < RGBA_BUFFER_COUNT; i++)
    {
        cuMemFree(m_rgbaBuffers[i]);
    }

    PROPAGATE_ERROR(m_rgbaOutputStream.destroy());

    return true;
}

bool CudaBayerDemosaicConsumer::threadShutdown()
{
    for (auto stream : m_streams) {
        PROPAGATE_ERROR(stream->shutdown());
    }

    for (auto stream : m_streams) {
        PROPAGATE_ERROR(stream->shutdownBeforePreview());
    }

    shutdownPreview();

    for (auto stream : m_streams) {
        PROPAGATE_ERROR(stream->shutdownAfterPreview());
    }

    PROPAGATE_ERROR(cleanupCUDA(&m_cudaContext));

    return true;
}

} // namespace ArgusSamples
