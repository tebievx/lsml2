$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result-title').hide();
    $('#result-text').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result-title').text('');
        $('#result-title').hide();
        $('#result-text').text('');
        $('#result-text').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        let form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            dataType: "json",
            success: function (data) {
                let task_id = data.task_id;

                if (task_id) {
                    let intervalID = setInterval(function() {
                            $.ajax({
                                type: 'GET',
                                dataType: "json",
                                url: `/task/${task_id}`,
                                async: false,
                                success: function (data) {
                                    if (data.ready) {
                                        clearInterval(intervalID);
                                        $('.loader').hide();
                                        $('#result-title').fadeIn(600);
                                        $('#result-title').text('Caption: ');
                                        $('#result-text').fadeIn(600);
                                        $('#result-text').text(data.result);
                                    }
                                }
                            });
                        },
                        200
                    )
                }

            }

        });

    });

});
