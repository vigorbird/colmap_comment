// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/controllers/incremental_mapper.h"

#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

namespace colmap {
namespace {

void IterativeGlobalRefinement(const IncrementalMapperOptions& options,
                               const IncrementalMapper::Options& mapper_options,
                               IncrementalMapper& mapper) {
  LOG(INFO) << "Retriangulation and Global bundle adjustment";
  mapper.IterativeGlobalRefinement(options.ba_global_max_refinements,
                                   options.ba_global_max_refinement_change,
                                   mapper_options,
                                   options.GlobalBundleAdjustment(),
                                   options.Triangulation());
  mapper.FilterImages(mapper_options);
}

void ExtractColors(const std::string& image_path,
                   const image_t image_id,
                   Reconstruction& reconstruction) {
  if (!reconstruction.ExtractColorsForImage(image_id, image_path)) {
    LOG(WARNING) << StringPrintf("Could not read image %s at path %s.",
                                 reconstruction.Image(image_id).Name().c_str(),
                                 image_path.c_str());
  }
}

void WriteSnapshot(const Reconstruction& reconstruction,
                   const std::string& snapshot_path) {
  LOG(INFO) << "Creating snapshot";
  // Get the current timestamp in milliseconds.
  const size_t timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  // Write reconstruction to unique path with current timestamp.
  const std::string path =
      JoinPaths(snapshot_path, StringPrintf("%010d", timestamp));
  CreateDirIfNotExists(path);
  VLOG(1) << "=> Writing to " << path;
  reconstruction.Write(path);
}

}  // namespace

IncrementalMapper::Options IncrementalMapperOptions::Mapper() const {
  IncrementalMapper::Options options = mapper;
  options.abs_pose_refine_focal_length = ba_refine_focal_length;
  options.abs_pose_refine_extra_params = ba_refine_extra_params;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.num_threads = num_threads;
  options.local_ba_num_images = ba_local_num_images;
  options.fix_existing_images = fix_existing_images;
  return options;
}

IncrementalTriangulator::Options IncrementalMapperOptions::Triangulation()
    const {
  IncrementalTriangulator::Options options = triangulation;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  return options;
}

BundleAdjustmentOptions IncrementalMapperOptions::LocalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = ba_local_function_tolerance;
  options.solver_options.gradient_tolerance = 10.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_local_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.logging_type = ceres::LoggingType::SILENT;
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = false;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.min_num_residuals_for_multi_threading =
      ba_min_num_residuals_for_multi_threading;
  options.loss_function_scale = 1.0;
  options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  return options;
}

BundleAdjustmentOptions IncrementalMapperOptions::GlobalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = ba_global_function_tolerance;
  options.solver_options.gradient_tolerance = 1.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_global_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
  options.solver_options.minimizer_progress_to_stdout = false;
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = false;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.min_num_residuals_for_multi_threading = ba_min_num_residuals_for_multi_threading;
  options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  return options;
}

bool IncrementalMapperOptions::Check() const {
  CHECK_OPTION_GT(min_num_matches, 0);
  CHECK_OPTION_GT(max_num_models, 0);
  CHECK_OPTION_GT(max_model_overlap, 0);
  CHECK_OPTION_GE(min_model_size, 0);
  CHECK_OPTION_GT(init_num_trials, 0);
  CHECK_OPTION_GT(min_focal_length_ratio, 0);
  CHECK_OPTION_GT(max_focal_length_ratio, 0);
  CHECK_OPTION_GE(max_extra_param, 0);
  CHECK_OPTION_GE(ba_local_num_images, 2);
  CHECK_OPTION_GE(ba_local_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_global_images_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_points_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_images_freq, 0);
  CHECK_OPTION_GT(ba_global_points_freq, 0);
  CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_local_max_refinements, 0);
  CHECK_OPTION_GE(ba_local_max_refinement_change, 0);
  CHECK_OPTION_GT(ba_global_max_refinements, 0);
  CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
  CHECK_OPTION_GE(snapshot_images_freq, 0);
  CHECK_OPTION(Mapper().Check());
  CHECK_OPTION(Triangulation().Check());
  return true;
}

//搜索 IncrementalMapperController构造函数
IncrementalMapperController::IncrementalMapperController(
    std::shared_ptr<const IncrementalMapperOptions> options,
    const std::string& image_path,
    const std::string& database_path,
    std::shared_ptr<ReconstructionManager> reconstruction_manager)
    : options_(std::move(options)),
      image_path_(image_path),
      database_path_(database_path),
      reconstruction_manager_(std::move(reconstruction_manager)) {
  THROW_CHECK(options_->Check());
  RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
  RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
  RegisterCallback(LAST_IMAGE_REG_CALLBACK);
}

void IncrementalMapperController::Run() {
  Timer run_timer;
  run_timer.Start();
  if (!LoadDatabase()) {
    return;
  }

  IncrementalMapper::Options init_mapper_options = options_->Mapper();
  Reconstruct(init_mapper_options);

  const size_t kNumInitRelaxations = 2;
  for (size_t i = 0; i < kNumInitRelaxations; ++i) {
    if (reconstruction_manager_->Size() > 0 || CheckIfStopped()) {
      break;
    }

    LOG(INFO) << "=> Relaxing the initialization constraints.";
    init_mapper_options.init_min_num_inliers /= 2;
    Reconstruct(init_mapper_options);

    if (reconstruction_manager_->Size() > 0 || CheckIfStopped()) {
      break;
    }

    LOG(INFO) << "=> Relaxing the initialization constraints.";
    init_mapper_options.init_min_tri_angle /= 2;
    Reconstruct(init_mapper_options);
  }

  run_timer.PrintMinutes();
}

bool IncrementalMapperController::LoadDatabase() {
  LOG(INFO) << "Loading database";

  // Make sure images of the given reconstruction are also included when
  // manually specifying images for the reconstruction procedure.
  std::unordered_set<std::string> image_names = options_->image_names;
  if (reconstruction_manager_->Size() == 1 && !options_->image_names.empty()) {
    const auto& reconstruction = reconstruction_manager_->Get(0);
    for (const image_t image_id : reconstruction->RegImageIds()) {
      const auto& image = reconstruction->Image(image_id);
      image_names.insert(image.Name());
    }
  }

  Database database(database_path_);
  Timer timer;
  timer.Start();
  const size_t min_num_matches = static_cast<size_t>(options_->min_num_matches);
  database_cache_ = DatabaseCache::Create(
      database, min_num_matches, options_->ignore_watermarks, image_names);
  timer.PrintMinutes();

  if (database_cache_->NumImages() == 0) {
    LOG(WARNING) << "No images with matches found in the database";
    return false;
  }

  return true;
}

IncrementalMapperController::Status
IncrementalMapperController::InitializeReconstruction(IncrementalMapper& mapper,
                                                      const IncrementalMapper::Options& mapper_options,
                                                      Reconstruction& reconstruction) {

  image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
  image_t image_id2 = static_cast<image_t>(options_->init_image_id2);

  // Try to find good initial pair.
  TwoViewGeometry two_view_geometry;
  if (!options_->IsInitialPairProvided()) {
    LOG(INFO) << "Finding good initial image pair";
    //整个代码就这里调用了FindInitialImagePair 函数
    const bool find_init_success = mapper.FindInitialImagePair( mapper_options, two_view_geometry, image_id1, image_id2);
    if (!find_init_success) {
      LOG(INFO) << "=> No good initial image pair found.";
      return Status::NO_INITIAL_PAIR;
    }
  } else {
    if (!reconstruction.ExistsImage(image_id1) ||
        !reconstruction.ExistsImage(image_id2)) {
      LOG(INFO) << StringPrintf("=> Initial image pair #%d and #%d do not exist.",
                                image_id1,
                                image_id2);
      return Status::BAD_INITIAL_PAIR;
    }
    const bool provided_init_success = mapper.EstimateInitialTwoViewGeometry(
        mapper_options, two_view_geometry, image_id1, image_id2);
    if (!provided_init_success) {
      LOG(INFO) << "Provided pair is insuitable for intialization.";
      return Status::BAD_INITIAL_PAIR;
    }
  }

  LOG(INFO) << StringPrintf("Initializing with image pair #%d and #%d", image_id1, image_id2);
  //2.非常重要的函数！
  mapper.RegisterInitialImagePair( mapper_options, two_view_geometry, image_id1, image_id2);

  //3.
  LOG(INFO) << "Global bundle adjustment";
  mapper.AdjustGlobalBundle(mapper_options, options_->GlobalBundleAdjustment());//

  //4.
  reconstruction.Normalize();//

  //5.
  mapper.FilterPoints(mapper_options);

  //6.
  mapper.FilterImages(mapper_options);

  // Initial image pair failed to register.
  if (reconstruction.NumRegImages() == 0 || reconstruction.NumPoints3D() == 0) {
    return Status::BAD_INITIAL_PAIR;
  }

  //默认会进入这个条件
  if (options_->extract_colors) {
    //7.
    ExtractColors(image_path_, image_id1, reconstruction);
  }
  return Status::SUCCESS;
}//end function InitializeReconstruction


bool IncrementalMapperController::CheckRunGlobalRefinement(
    const Reconstruction& reconstruction,
    const size_t ba_prev_num_reg_images,
    const size_t ba_prev_num_points) {
  return reconstruction.NumRegImages() >= options_->ba_global_images_ratio * ba_prev_num_reg_images ||
         reconstruction.NumRegImages() >= options_->ba_global_images_freq + ba_prev_num_reg_images ||
         reconstruction.NumPoints3D() >=  options_->ba_global_points_ratio * ba_prev_num_points ||
         reconstruction.NumPoints3D() >=  options_->ba_global_points_freq + ba_prev_num_points;
}


IncrementalMapperController::Status
IncrementalMapperController::ReconstructSubModel(
    IncrementalMapper& mapper,
    const IncrementalMapper::Options& mapper_options,
    const std::shared_ptr<Reconstruction>& reconstruction) {

  //1.主要是读取图像和特征点数据，并获取一共有多少个相机
  mapper.BeginReconstruction(reconstruction);

  ////////////////////////////////////////////////////////////////////////////
  // Register initial pair
  ////////////////////////////////////////////////////////////////////////////

  if (reconstruction->NumRegImages() == 0) {
    //整个代码就这里调用了InitializeReconstruction函数
    const Status init_status = IncrementalMapperController::InitializeReconstruction( mapper, mapper_options, *reconstruction);
    if (init_status != Status::SUCCESS) {
      return init_status;
    }
  }
  Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);//好像在c++代码中这是一个空函数！！

  ////////////////////////////////////////////////////////////////////////////
  // Incremental mapping
  ////////////////////////////////////////////////////////////////////////////

  size_t snapshot_prev_num_reg_images = reconstruction->NumRegImages();
  size_t ba_prev_num_reg_images = reconstruction->NumRegImages();
  size_t ba_prev_num_points = reconstruction->NumPoints3D();

  bool reg_next_success = true;
  bool prev_reg_next_success = true;
  //注意这是一个大的for循环
  do {
    if (CheckIfStopped()) {
      break;
    }

    prev_reg_next_success = reg_next_success;
    reg_next_success = false;
    //2.找到下一步的所有可能的要被注册的图像，而不是只找一个图像, 并且返回的图像根据特征点的分布得分降序排序
    const std::vector<image_t> next_images =  mapper.FindNextImages(mapper_options);

    if (next_images.empty()) {
      break;
    }

    image_t next_image_id;
    for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
      next_image_id = next_images[reg_trial];
      const Image& next_image = reconstruction->Image(next_image_id);

      LOG(INFO) << StringPrintf("Registering image #%d (%d)",
                                next_image_id,
                                reconstruction->NumRegImages() + 1);
      LOG(INFO) << StringPrintf("=> Image sees %d / %d points",
                                next_image.NumVisiblePoints3D(),
                                next_image.NumObservations());
      //3.计算图像的位姿
      reg_next_success =  mapper.RegisterNextImage(mapper_options, next_image_id);
      //需要注意只要注册成功一个图像，立刻退出这个循环for
      if (reg_next_success) {
        break;
      } else {
        LOG(INFO) << "=> Could not register, trying another image.";

        // If initial pair fails to continue for some time,
        // abort and try different initial pair.
        const size_t kMinNumInitialRegTrials = 30;
        if (reg_trial >= kMinNumInitialRegTrials &&
             reconstruction->NumRegImages() < static_cast<size_t>(options_->min_model_size)) {//min_model_size 默认值 = 30
          break;
        }
      }
    }//遍历完所有的要注册的图像

    if (reg_next_success) {
      mapper.TriangulateImage(options_->Triangulation(), next_image_id);
      mapper.IterativeLocalRefinement(options_->ba_local_max_refinements,
                                      options_->ba_local_max_refinement_change,
                                      mapper_options,
                                      options_->LocalBundleAdjustment(),
                                      options_->Triangulation(),
                                      next_image_id);

      if (CheckRunGlobalRefinement(*reconstruction, ba_prev_num_reg_images, ba_prev_num_points)) {
        IterativeGlobalRefinement(*options_, mapper_options, mapper);
        ba_prev_num_points = reconstruction->NumPoints3D();
        ba_prev_num_reg_images = reconstruction->NumRegImages();
      }

      if (options_->extract_colors) {
        ExtractColors(image_path_, next_image_id, *reconstruction);
      }

      if (options_->snapshot_images_freq > 0 &&
          reconstruction->NumRegImages() >= options_->snapshot_images_freq + snapshot_prev_num_reg_images) {
        snapshot_prev_num_reg_images = reconstruction->NumRegImages();
        WriteSnapshot(*reconstruction, options_->snapshot_path);
      }

      Callback(NEXT_IMAGE_REG_CALLBACK);
    }

    const size_t max_model_overlap =  static_cast<size_t>(options_->max_model_overlap);
    if (mapper.NumSharedRegImages() >= max_model_overlap) {
      break;
    }

    // If no image could be registered, try a single final global iterative
    // bundle adjustment and try again to register one image. If this fails
    // once, then exit the incremental mapping.
    if (!reg_next_success && prev_reg_next_success) {
      IterativeGlobalRefinement(*options_, mapper_options, mapper);
    }
  } while (reg_next_success || prev_reg_next_success);
  //退出大循环！！！！！！！！！


  if (CheckIfStopped()) {
    return Status::INTERRUPTED;
  }

  // Only run final global BA, if last incremental BA was not global.
  if (reconstruction->NumRegImages() >= 2 &&
      reconstruction->NumRegImages() != ba_prev_num_reg_images &&
      reconstruction->NumPoints3D() != ba_prev_num_points) {
    IterativeGlobalRefinement(*options_, mapper_options, mapper);
  }
  return Status::SUCCESS;
}//end function ReconstructSubModel！




void IncrementalMapperController::Reconstruct(const IncrementalMapper::Options& mapper_options) {
  IncrementalMapper mapper(database_cache_);//小构造函数就是初始赋值一些变量

  // Is there a sub-model before we start the reconstruction? I.e. the user
  // has imported an existing reconstruction.
  const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
  THROW_CHECK_LE(reconstruction_manager_->Size(), 1) << "Can only resume from a "
                                                        "single reconstruction, but "
                                                        "multiple are given.";

  //init_num_trials默认值是200
  for (int num_trials = 0; num_trials < options_->init_num_trials; ++num_trials) {
    if (CheckIfStopped()) {
      break;
    }
    size_t reconstruction_idx;
    if (!initial_reconstruction_given || num_trials > 0) {
      reconstruction_idx = reconstruction_manager_->Add();//搜索 ReconstructionManager::Add()
    } else {
      reconstruction_idx = 0;
    }
    std::shared_ptr<Reconstruction> reconstruction =  reconstruction_manager_->Get(reconstruction_idx);

    const Status status = ReconstructSubModel(mapper, mapper_options, reconstruction);//非常重要的函数！！！！

    switch (status) {
      case Status::INTERRUPTED:
        mapper.EndReconstruction(/*discard=*/false);
        return;

      case Status::NO_INITIAL_PAIR:
      case Status::BAD_INITIAL_PAIR:
        mapper.EndReconstruction(/*discard=*/true);
        reconstruction_manager_->Delete(reconstruction_idx);
        // If both initial images are manually specified, there is no need for
        // further initialization trials.
        if (options_->IsInitialPairProvided()) {
          return;
        }
        break;

      case Status::SUCCESS: {
          // Remember the total number of registered images before potentially
          // discarding it below due to small size, so we can out of the main
          // loop, if all images were registered.
          const size_t total_num_reg_images = mapper.NumTotalRegImages();

          // If the total number of images is small then do not enforce the
          // minimum model size so that we can reconstruct small image
          // collections. Always keep the first reconstruction, independent of
          // size.
          const size_t min_model_size = std::min<size_t>( 0.8 * database_cache_->NumImages(), options_->min_model_size);
          if ((options_->multiple_models && 
              reconstruction_manager_->Size() > 1 &&
              reconstruction->NumRegImages() < min_model_size) ||
              reconstruction->NumRegImages() == 0) {

              mapper.EndReconstruction(/*discard=*/true);
              reconstruction_manager_->Delete(reconstruction_idx);

          } else {
            mapper.EndReconstruction(/*discard=*/false);
          }

          Callback(LAST_IMAGE_REG_CALLBACK);

          //非常重要！优化结束
          if (initial_reconstruction_given || //用户是否给定了一个初始的structure
              !options_->multiple_models ||  //默认值是true
              reconstruction_manager_->Size() >= static_cast<size_t>(options_->max_num_models) || //submodel的数量足够多， max_num_models默认值 = 50
              total_num_reg_images >= database_cache_->NumImages() - 1) {//如果所有的图像全部都被注册过了
            return;
          }
      } break;

      default:
        LOG(FATAL_THROW) << "Unknown reconstruction status.";
    }//end switch case
  }//end for 遍历次数！
}//end function Reconstruct!!!



void IncrementalMapperController::TriangulateReconstruction(
    const std::shared_ptr<Reconstruction>& reconstruction) {
  THROW_CHECK(LoadDatabase());
  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction);

  LOG(INFO) << "Iterative triangulation";
  const std::vector<image_t>& reg_image_ids = reconstruction->RegImageIds();
  for (size_t i = 0; i < reg_image_ids.size(); ++i) {
    const image_t image_id = reg_image_ids[i];
    const auto& image = reconstruction->Image(image_id);

    LOG(INFO) << StringPrintf("Triangulating image #%d (%d)", image_id, i);
    const size_t num_existing_points3D = image.NumPoints3D();
    LOG(INFO) << "=> Image sees " << num_existing_points3D << " / "
              << image.NumObservations() << " points";

    mapper.TriangulateImage(options_->Triangulation(), image_id);
    VLOG(1) << "=> Triangulated "
            << (image.NumPoints3D() - num_existing_points3D) << " points";
  }

  LOG(INFO) << "Retriangulation and Global bundle adjustment";
  mapper.IterativeGlobalRefinement(options_->ba_global_max_refinements,
                                   options_->ba_global_max_refinement_change,
                                   options_->Mapper(),
                                   options_->GlobalBundleAdjustment(),
                                   options_->Triangulation(),
                                   /*normalize_reconstruction=*/false);
  mapper.EndReconstruction(/*discard=*/false);

  LOG(INFO) << "Extracting colors";
  reconstruction->ExtractColorsForAllImages(image_path_);
}

}  // namespace colmap
