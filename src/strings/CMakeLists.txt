set(strings_fppFiles
    stdlib_base64_encode.fypp
    stdlib_base64_decode.fypp
    stdlib_base64.fypp
    stdlib_string_type.fypp
    stdlib_string_type_constructor.fypp
    stdlib_str2num.fypp
    stdlib_strings_to_string.fypp
    stdlib_strings.fypp
    )

set(strings_cppFiles
    )

set(strings_f90Files
    )

configure_stdlib_target(${PROJECT_NAME}_strings strings_f90Files strings_fppFiles strings_cppFiles)

target_link_libraries(${PROJECT_NAME}_strings PUBLIC ${PROJECT_NAME}_core)

if(STDLIB_BASE64_USE_AVX2_BINDING)
    target_sources(${PROJECT_NAME}_strings PRIVATE
        base64/avx2_base64.cpp
    )

    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/base64/avx2_base64.cpp
        PROPERTIES COMPILE_OPTIONS "-O3;-mavx2;-mbmi2"
    )
endif()
