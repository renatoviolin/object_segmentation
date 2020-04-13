jQuery(document).ready(function () {

    $('#btn-process').on('click', function () {
        var form_data = new FormData();
        form_data.append('file', $('#input_file').prop('files')[0]);

        $.ajax({
            url: '/generate',
            type: "post",
            contentType: "application/json",
            data: form_data,
            processData: false,
            contentType: false,
            cache: false,
            beforeSend: function () {
                $('.header_label').hide()
                $('.img').hide()
                $('.div-img-container-resnet').html('')
                $('.div-img-container-xception').html('')
                $('.div-img-container-centermask').html('')
                $('.overlay').show()
            },
            complete: function () {
                $('.overlay').hide()
                $('.header_label').show()
                $('.img').show()
            }
        }).done(function (jsondata, textStatus, jqXHR) {
            $('#img_input').attr('src', jsondata['input_file'])

            console.log(jsondata)
            for (i = 0; i < jsondata['segment_resnet'].length; i++) {
                titulo = jsondata['classes_xception'][i]
                url = jsondata['segment_resnet'][i]
                img_html = `
                    <div class="preview">
                        <span class="class-name">${titulo}</span>
                        <img src='${url}' class="img">
                    </div>
                `
                $('.div-img-container-resnet').append(img_html)
            }

            for (i = 0; i < jsondata['segment_xception'].length; i++) {
                titulo = jsondata['classes_xception'][i]
                url = jsondata['segment_xception'][i]
                img_html = `
                    <div class="preview">
                        <span class="class-name">${titulo}</span>
                        <img src='${url}' class="img">
                    </div>
                `
                $('.div-img-container-xception').append(img_html)
            }

            url = jsondata['segment_centermask']
            img_html = `
                <div class="centermask">
                    <img src='${url}' class="img-centermask">
                </div>
            `
            $('.div-img-container-centermask').append(img_html)



        }).fail(function (jsondata, textStatus, jqXHR) {
            alert(jsondata['responseJSON'])
        });
    })

})