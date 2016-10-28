!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Fortran version block finding for csr type matrix.
!Complexity: sum over p**2, p is the collection of block size.
!Author: Leo
!Data: Oct. 8. 2015
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!Get the specific sub-block in 2-site vmps update
!
!Parameters
!-------------------------
!fl,o1,o2,fr: ndarray(n=3,4,4,3).
!    the decoupled tensor form.
!indices: 2d array.
!    the desired indices for 'down' axis with size - (nl,hndim,hndim,nr).
!
!Return:
!--------------------------
!res: matrix,
subroutine fget_subblock2a(fl,o1,o2,fr,indices,res,nl,nr,nhl,nhc,nhr,hndim,ndim)
    implicit none
    integer,intent(in) :: nl,nr,ndim,nhl,nhr,nhc,hndim
    integer,intent(in) :: indices(ndim,4)
    complex*16,intent(in) :: fl(nl,nhl,nl),fr(nr,nhr,nr),o1(nhl,hndim,hndim,nhc),o2(nhc,hndim,hndim,nhr)
    complex*16,intent(out) :: res(ndim,ndim)
    integer :: i,j
    complex*16 :: col_temp_fl(nl,nhl),col_temp_fr(nr,nhr),col_temp_o1(nhl,hndim,nhc),col_temp_o2(nhc,hndim,nhr),&
        temp_fl(nhl),temp_fr(nhr),temp_o1(nhl,nhc),temp_o2(nhc,nhr),temp_c1(1,nhc),temp_c2(1,nhr)
    
    !f2py intent(in) :: fl,o1,o2,fr,indices,nl,nr,nhl,nhc,nhr,hndim,ndim
    !f2py intent(out) :: res

    !prepair datas
    do j=1,ndim
        !cache datas
        col_temp_fl=fl(:,:,indices(j,1)+1)
        col_temp_o1=o1(:,:,indices(j,2)+1,:)
        col_temp_o2=o2(:,:,indices(j,3)+1,:)
        col_temp_fr=fr(:,:,indices(j,4)+1)
        do i=1,ndim
            !cache datas
            temp_fl=col_temp_fl(indices(i,1)+1,:)
            temp_o1=col_temp_o1(:,indices(i,2)+1,:)
            temp_o2=col_temp_o2(:,indices(i,3)+1,:)
            temp_fr=col_temp_fr(indices(i,4)+1,:)
            !do the contraction
            temp_c1=matmul(reshape(temp_fl,[1,nhl]),temp_o1)
            temp_c2=matmul(temp_c1,temp_o2)
            res(i,j)=sum(temp_c2(1,:)*temp_fr)
        enddo
    enddo
end subroutine fget_subblock2a


subroutine fget_subblock2b(fl,o1,o2,fr,indices,res,nl,nr,nhl,nhc,nhr,hndim,ndim)
    implicit none
    integer,intent(in) :: nl,nr,ndim,nhl,nhr,nhc,hndim
    integer,intent(in) :: indices(ndim,4)
    complex*16,intent(in) :: fl(nl,nhl,nl),fr(nr,nhr,nr),o1(nhl,hndim,hndim,nhc),o2(nhc,hndim,hndim,nhr)
    complex*16,intent(out) :: res(ndim,ndim)
    integer :: j,cind(4),i
    complex*16 :: col_temp_fl(nl,nhl),col_temp_fr(nr,nhr),col_temp_o1(nhl,hndim,nhc),col_temp_o2(nhc,hndim,nhr),&
        temp_left(nl*hndim*hndim,nhr),temp_center(nhl*hndim,hndim*nhr),temp(nl,hndim,hndim,nr)
    
    !f2py intent(in) :: fl,o1,o2,fr,indices,nl,nr,nhl,nhc,nhr,hndim,ndim
    !f2py intent(out) :: res

    !prepair datas
    do j=1,ndim
        !cache datas
        col_temp_fl=fl(:,:,indices(j,1)+1)
        col_temp_o1=o1(:,:,indices(j,2)+1,:)
        col_temp_o2=o2(:,:,indices(j,3)+1,:)
        col_temp_fr=fr(:,:,indices(j,4)+1)

        !first, contract center blocks.
        temp_center=matmul(reshape(col_temp_o1,[nhl*hndim,nhc]),reshape(col_temp_o2,[nhc,hndim*nhr]))
        temp_left=reshape(matmul(col_temp_fl,reshape(temp_center,[nhl,hndim*hndim*nhr])),[nl*hndim*hndim,nhr])
        !sencond, contract left blocks.
        temp=reshape(matmul(temp_left,transpose(col_temp_fr)),[nl,hndim,hndim,nr])
        do i=1,ndim
            cind=indices(i,:)+1
            res(i,j)=temp(cind(1),cind(2),cind(3),cind(4))
        enddo
    enddo
end subroutine fget_subblock2b


!is_identity: 0 -> no, 1 -> left, 2 -> right
subroutine fget_subblock_dmrg(hl,hr,indices,ndim,nl,nr,is_identity,res)
    implicit none
    integer,intent(in) :: indices(ndim,2),ndim,nl,nr,is_identity
    complex*16,intent(in) :: hl(nl,nl),hr(nr,nr)
    complex*16,intent(out) :: res(ndim,ndim)
    integer :: i,j,il,ir,jl,jr
    complex*16 :: temp_cl(nl),temp_cr(nr)
    
    !f2py intent(in) :: hl,hr,indices,ndim,nl,nr,is_identity
    !f2py intent(out) :: res

    if(is_identity==0) then
        do j=1,ndim
            jl=indices(j,1)+1
            jr=indices(j,2)+1
            temp_cl=hl(:,jl)
            temp_cr=hr(:,jr)
            do i=1,ndim
                il=indices(i,1)+1
                ir=indices(i,2)+1
                res(i,j)=temp_cl(il)*temp_cr(ir)
            enddo
        enddo
    else if(is_identity==1) then
        do j=1,ndim
            jl=indices(j,1)+1
            jr=indices(j,2)+1
            temp_cr=hr(:,jr)
            do i=1,ndim
                il=indices(i,1)+1
                ir=indices(i,2)+1
                if(il==jl) then
                    res(i,j)=temp_cr(ir)
                endif
            enddo
        enddo
    else
        do j=1,ndim
            jl=indices(j,1)+1
            jr=indices(j,2)+1
            temp_cl=hl(:,jl)
            do i=1,ndim
                il=indices(i,1)+1
                ir=indices(i,2)+1
                if(ir==jr) then
                    res(i,j)=temp_cl(il)
                endif
            enddo
        enddo
    endif
end subroutine fget_subblock_dmrg

subroutine fget_subblock1(fl,o1,fr,indices,res,nl,nr,nhl,nhr,hndim,ndim)
    implicit none
    integer,intent(in) :: nl,nr,ndim,nhl,nhr,hndim
    integer,intent(in) :: indices(ndim,3)
    complex*16,intent(in) :: fl(nl,nhl,nl),fr(nr,nhr,nr),o1(nhl,hndim,hndim,nhr)
    complex*16,intent(out) :: res(ndim,ndim)
    integer :: j,cind(3),i
    complex*16 :: col_temp_fl(nl,nhl),col_temp_fr(nr,nhr),col_temp_o1(nhl,hndim,nhr),&
        temp_left(nl*hndim,nhr),temp(nl,hndim,nr)
    
    !f2py intent(in) :: fl,o1,o2,fr,indices,nl,nr,nhl,nhr,hndim,ndim
    !f2py intent(out) :: res

    !prepair datas
    do j=1,ndim
        !cache datas
        col_temp_fl=fl(:,:,indices(j,1)+1)
        col_temp_o1=o1(:,:,indices(j,2)+1,:)
        col_temp_fr=fr(:,:,indices(j,3)+1)

        !first, contract center blocks.
        temp_left=reshape(matmul(col_temp_fl,reshape(col_temp_o1,[nhl,hndim*nhr])),[nl*hndim,nhr])
        !sencond, contract left blocks.
        temp=reshape(matmul(temp_left,transpose(col_temp_fr)),[nl,hndim,nr])
        do i=1,ndim
            cind=indices(i,:)+1
            res(i,j)=temp(cind(1),cind(2),cind(3))
        enddo
    enddo
end subroutine fget_subblock1


!take only the values in arr that are in diresired indices.
subroutine ftake_only(arr,indices,ndim,ndim0,mask1,mask2)
    implicit none
    integer,intent(in) :: arr(ndim0),indices(ndim),ndim0,ndim
    logical,intent(out) :: mask1(ndim0),mask2(ndim)
    integer :: i,j,ai,iind,k
    
    !f2py intent(in) :: arr,indices,ndim,ndim0
    !f2py intent(out) :: mask1,mask2

    mask1=.false.
    mask2=.false.
    i=1
    j=1
    ai=arr(i)
    iind=indices(j)
    do k=1,ndim0+ndim
        if(ai==iind) then
            mask2(j)=.true.
            mask1(i)=.true.
            j=j+1
            i=i+1
            if(i>ndim0 .or. j>ndim0) then
                exit
            endif
            ai=arr(i)
            iind=indices(j)
        else if(ai>iind) then
            j=j+1
            if(j>ndim) then
                exit
            endif
            iind=indices(j)
        else if(ai<iind) then
            i=i+1
            if(i>ndim0) then
                exit
            endif
            ai=arr(i)
        endif
    enddo
end subroutine ftake_only

